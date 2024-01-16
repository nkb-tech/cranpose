from .utils import (reshape_features, convert_locations_to_kpts,
                    convert_locations_to_boxes, center_form_to_corner_form,
                    detect_center_and_corners)

# Detector

def pred_detector_features(model, image_torch, priors, w, h, scale, center_variance, size_variance):
    stages_output = model(image_torch)
        
    init_masks_pred, grid_confidences_pred, grid_locations_pred, grid_vals_pred = stages_output[0:4]

    # corners
    grid_priors = priors['grid_priors_list'][0]
    
    grid_confidences_pred = reshape_features(grid_confidences_pred)
    grid_locations_pred = reshape_features(grid_locations_pred)
    grid_vals_pred = reshape_features(grid_vals_pred)
    grid_confidences_pred = F.softmax(grid_confidences_pred, dim=2)
    
    grid_kpts_pred= convert_locations_to_kpts(grid_locations_pred, grid_priors, center_variance=center_variance)
    grid_kpts_pred[...,0] *=w/scale
    grid_kpts_pred[...,1] *=h/scale

    
    grid_confidences_pred_np = grid_confidences_pred.cpu().data.numpy()
    grid_vals_pred_np = grid_vals_pred.cpu().data.numpy()
    grid_kpts_pred_np = grid_kpts_pred.cpu().data.numpy()



    # center and bbox
    ssd_confidences_pred, ssd_locations_pred, ssd_vals_pred = stages_output[4:7]
    

    ssd_confidences_pred =reshape_features(ssd_confidences_pred)
    ssd_locations_pred =reshape_features(ssd_locations_pred)
    ssd_vals_pred =reshape_features(ssd_vals_pred)

    ssd_confidences_pred = F.softmax(ssd_confidences_pred, dim=2)

    ssd_boxes_pred = convert_locations_to_boxes(
        ssd_locations_pred, priors['ssd_priors'], center_variance, size_variance
    )
    ssd_boxes_pred = center_form_to_corner_form(ssd_boxes_pred)
    ssd_boxes_pred[...,::2] *= w/scale
    ssd_boxes_pred[...,1::2] *= h/scale

    return grid_confidences_pred_np, grid_vals_pred_np, grid_kpts_pred_np, ssd_confidences_pred, ssd_vals_pred, ssd_boxes_pred


# FIXME rename to "DeepDetector"
class StagDetector:
    """
    Class to detect marker corners
    """
    def __init__(self,
                 predictor,
                 min_center_score=0.3,
                 min_corner_score=0.3,
                 is_allow_no_corner_refine=True,
                 is_with_corner_refine=True):

        self.image_in = None
        self.tag_res = []
        self.predictor = predictor
        self.min_center_score = min_center_score
        self.min_corner_score = min_corner_score
        self.is_allow_no_corner_refine = is_allow_no_corner_refine
        self.is_with_corner_refine = is_with_corner_refine


    def detect_rois(self, image_in, scale = 1):

        min_center_score = self.min_center_score
        min_corner_score = self.min_corner_score
        is_allow_no_corner_refine = self.is_allow_no_corner_refine
        is_with_corner_refine = self.is_with_corner_refine
        self.cnn_timing = -1
        self.total_timing = -1
        self.predition_timing = -1

        # t0 = time.time() # timing
        
        #  pred features       
        features_pred = self.predictor.predict(image_in, scale)[0]
        self.cnn_timing = self.predictor.cnn_timing
        self.predition_timing = time.time() - t0

        # sparse unordered corners, bbox with 5 anchors (center + 4 ordered corners)
        image_res = detect_center_and_corners(features_pred, min_center_score = min_center_score, min_corner_score = min_corner_score)
        

        # center_to_corner_links and tags (4 ordered corners)
        tag_res, center_to_corner_links = center_and_corners_to_tags(image_res, is_allow_no_corner_refine=is_allow_no_corner_refine, is_with_corner_refine = is_with_corner_refine)
        # self.total_timing = time.time() -t0 # timing

        # save results
        self.image_res =image_res
        self.tag_res = tag_res 
        self.center_to_corner_links = center_to_corner_links
        self.features_pred = features_pred
        return tag_res


class DetectorPredictor:
    def __init__(self,
                 model,
                 device,
                 stride_list=[8, 16, 32, 64],
                 center_variance=0.1,
                 size_variance=0.2):
        model.eval()

        self.device = device
        self.model = model
        self.stride_list = stride_list
        self.priors = None
        self.center_variance = center_variance
        self.size_variance = size_variance

    def get_grid_and_ssd_priors(self, image_size):
        stride_list = self.stride_list
        device = self.device

        grid_priors_list = [
            generate_grid_priors(
                image_size,
                stride,
                clamp=False
                ).to(device).unsqueeze(0) for stride in stride_list]
        box_specs = get_box_specs(image_size,stride_list)
        ssd_priors = generate_ssd_priors(box_specs, image_size).to(device).unsqueeze(0)
        ssd_priors_corner_form = center_form_to_corner_form(ssd_priors)

        priors = {}
        priors['image_size'] = image_size
        priors['ssd_priors'] = ssd_priors
        priors['box_specs'] = box_specs
        priors['grid_priors_list'] = grid_priors_list
        priors['ssd_priors_corner_form'] = ssd_priors_corner_form

        return priors

    def _predict_embeddings(self, images, scale):
        device = self.device
        center_variance = self.center_variance
        size_variance = self.size_variance

        if type(images) is not list:
            images = [images]

        images = [
            crop_or_pad(image_in, scale, stride=max(self.stride_list))
            for image_in in images]


        if self.priors is None or self.priors['image_size'][0]!= images[0].shape[0] or self.priors['image_size'][1] != images[0].shape[1]:
            self.priors = self.get_grid_and_ssd_priors(images[0].shape[:2])

         # stage-1 prediction
        h,w = self.priors['image_size']

        image_torch = np_to_torch(images, device, std = 255)


        t0 = time.time()
        

        grid_confidences_pred_np, grid_vals_pred_np, grid_kpts_pred_np, ssd_confidences_pred, ssd_vals_pred, ssd_boxes_pred = pred_detector_features(self.model, image_torch, self.priors, w, h, scale, center_variance, size_variance)

        self.cnn_timing = time.time()-t0

        ssd_vals_pred_np = ssd_vals_pred.cpu().data.numpy()
        ssd_confidences_pred_np = ssd_confidences_pred.cpu().data.numpy()
        ssd_boxes_pred_np = ssd_boxes_pred.cpu().data.numpy()



        features_pred = []
        for ii in range(len(images)):
            features = { 'grid_pred':[grid_confidences_pred_np[ii,...], grid_kpts_pred_np[ii,...], grid_vals_pred_np[ii,...]],
                'ssd_pred':[ssd_confidences_pred_np[ii,...], ssd_boxes_pred_np[ii,...], ssd_vals_pred_np[ii,...]]}
            features_pred.append(features)


        return features_pred