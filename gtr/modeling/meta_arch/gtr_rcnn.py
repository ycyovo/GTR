import cv2
import torch
import torchvision
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances
from detectron2.data import transforms as T
from gtr.data.custom_build_augmentation import build_custom_augmentation

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .custom_rcnn import CustomRCNN
from .visualization import plot_tracking
from ..roi_heads.custom_fast_rcnn import custom_fast_rcnn_inference

from torch import nn
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from tqdm.auto import tqdm


def fuck_check( track_ids ) :
    for i in range( len(track_ids) ) :
        for j in range( len(track_ids) ) :
            if i != j and track_ids[i] == track_ids[j] :
                return True 
    return False

def correct_train( condition_final , matching_final ) :
    outer_row = False
    for row in range( 80 ) : 
        one_bool = matching_final[0][0][row].eq(1)
        num = one_bool.sum()
        if num > 1 :
            import ipdb; ipdb.set_trace()
            print(f" More than one matching on row: {row}")
            return False
        elif num == 0 :
            if outer_row :
                outer_row = False
            continue 
        else :
            if outer_row :
                import ipdb; ipdb.set_trace()
                print(f"Outer error on row:{row}")
                return False
        one_indices = torch.nonzero( one_bool ).squeeze().item()
        if condition_final[0][0][row].max() - condition_final[0][0][row][one_indices] > 0      :
            import ipdb; ipdb.set_trace()
            print(f"Maybe wrong position{row}?")
            return False
    return True


def box_post_precess( boxes , shape_ori , shape_tar ) :
    h_ori = shape_ori[0]
    w_ori = shape_ori[1]
    h_tar = shape_tar[0]
    w_tar = shape_tar[1]
    scale_h = h_tar / h_ori 
    scale_w = w_tar / w_ori 
    tensor_this = boxes.tensor.clone()
    tensor_this[:,1] *= scale_h
    tensor_this[:,3] *= scale_h
    tensor_this[:,0] *= scale_w
    tensor_this[:,2] *= scale_w
    return Boxes(tensor_this)

def output_model( model ) :
    for name, param in model.named_parameters():
        print(name, param.size())
        
def make_matrix_decrease( Matrix , dts ) :
    Mat_new = Matrix.clone()
    dts_copy = dts.clone()
    length_diff = Mat_new[0][0][0].shape[0] - dts.shape[0]
    if( length_diff >= 0 ) :
        padding = torch.ones(length_diff, dtype=Mat_new.dtype).to('cuda')
        dts_copy = torch.cat((dts_copy, padding), dim=0)

    for i in range(Mat_new.shape[2]) :
        Mat_new[0][0][i] *= dts_copy
    return Mat_new

def get_decrease_matrix(instance_list) :
    N_list = torch.tensor([ len(x) for x in instance_list ]) 
    tensors = []
    for i,x in enumerate(N_list) :
        if i != len(N_list) - 1 :
            this_tensor = torch.full( (x,) , 0.5*(0.9**(len(N_list)-i-2)) + 0.5 )
            tensors.append( this_tensor )
    dts = torch.cat(tensors).to('cuda')
    return dts

def get_distance_IOU_matrix(instance_list, Mat_n, Mat_Tn):

    # 
    # box_list = torch.cat([ x.get("gt_boxes").tensor for x in instance_list ])
    # Iou = torchvision.ops.generalized_box_iou(box_list,box_list) + 1
    # Iou_extense = torch.nn.functional.pad( Iou , ( 0 , N_diff - len(box_list) , 0 , N_diff - len(box_list) ) )

    N_list = torch.tensor([ len(x) for x in instance_list ]) 
    N_presum_list = torch.cumsum( N_list , dim=0 )


    instance_list_n = [instance_list[-1]]
    instance_list_Tn = instance_list[:-1]
    box_list_n = torch.cat([ x.get("gt_boxes").tensor for x in instance_list_n ])
    box_list_Tn = torch.cat([ x.get("gt_boxes").tensor for x in instance_list_Tn ])
    Iou = torchvision.ops.generalized_box_iou(box_list_n,box_list_Tn) + 1

    Iou_extense = torch.nn.functional.pad( Iou , ( 0 , Mat_Tn - len(box_list_Tn) , 0 , Mat_n - len(box_list_n) ) )

    return Iou_extense.unsqueeze(0).unsqueeze(0)

    ### To be completed : fading coefficient

def get_outdated(instance_list,flag,frame_st):
    my_dict = {}
    cur_id = 0
    new_id_cnt = 0
    old_id_map = {}
    new_id_map = {}
    is_outdated = [False]*1664
    for frame_id,x in enumerate( instance_list ) :
        if flag : 
            track_ids = x.get('gt_instance_ids')
        else :
            track_ids = x['instances'].get('track_ids')

        for i,value_tensor in enumerate( track_ids ) :
            old_id_map[ cur_id ] = (frame_id+frame_st,i)
            value = value_tensor.item()
            last_id = my_dict.get( value )
            if last_id == None :
                my_dict[value] = cur_id
            else :
                is_outdated[last_id] = True
                my_dict[value] = cur_id
            cur_id += 1

    for i in range( cur_id ) :
        if not is_outdated[i] :
            new_id_map[new_id_cnt] = old_id_map[i]
            new_id_cnt += 1

    for i in range( cur_id , 1664 ) :
        is_outdated[i] = True
    
    return is_outdated , new_id_map



def get_matching_matrix(instance_list, Mat_n ,  Mat_Tn):
    # import ipdb; ipdb.set_trace()
    # if N_list.sum().item() > self.fuck_max :
    #     self.fuck_max = N_list.sum().item()
    #     print("New_max : " , self.fuck_max)
    #     print(N_list)
    #     self.flag = True
    T = len(instance_list)
    N_list = torch.tensor([ len(x) for x in instance_list ]) 
    N_presum_list = torch.cumsum( N_list , dim=0 )
    my_dict = {}

    # print(N_list)
    # print(N_presum_list)
    # n * Tn
    matching_matrix = torch.zeros( ( Mat_n, Mat_Tn ) )
    cur_id = 0  
    for frame_id,x in enumerate(instance_list) :
        track_id = x.get("gt_instance_ids")
        if frame_id == len(instance_list) - 1 :
            cur_id = 0
        for i,value_tensor in enumerate(track_id):
            value = value_tensor.item()
            last_id = my_dict.get(value)
            if last_id == None:
                my_dict[value] = cur_id
            else :
                if frame_id == len(instance_list) - 1 : 
                    # matching_matrix[last_id][cur_id] = 1
                    matching_matrix[cur_id][last_id] = 1       
                my_dict[value] = cur_id
            cur_id += 1
    
    
    # Tn * Tn
    # matching_matrix = torch.zeros( (N_diff,N_diff) )
    # cur_id = 0
    # for frame_id,x in enumerate(instance_list) :
    #     track_id = x.get("gt_instance_ids")
    #     for i,value_tensor in enumerate(track_id):
    #         value = value_tensor.item()
    #         last_id = my_dict.get(value)
    #         if last_id == None:
    #             my_dict[value] = cur_id
    #         else :
    #             matching_matrix[last_id][cur_id] = 1
    #             matching_matrix[cur_id][last_id] = 1
    #             my_dict[value] = cur_id
    #         cur_id += 1

    return ((matching_matrix.unsqueeze(0).unsqueeze(0))*2-1).to('cuda') , N_presum_list[-2]


# def ycy_code(instance_list, is_gt):

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=10, class_emb_size=1):
        super().__init__()
        
        # The embedding layer will map the class label to a vector of size class_emb_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=80,           # the target image resolution
            in_channels=1 + class_emb_size, # Additional input channels for class cond.
            out_channels=1,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64), 
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
            ),
        )
    # Our forward method now takes the class labels as an additional argument
    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        # class conditioning in right shape to add as additional input channels
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, class_labels), 1) # (bs, 2, 52, 80)
        return self.model(net_input, t).sample # (bs, 2, 52, 80)


@META_ARCH_REGISTRY.register()
class GTRRCNN(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        """
        """
        self.test_len = kwargs.pop('test_len')
        self.overlap_thresh = kwargs.pop('overlap_thresh')
        self.min_track_len = kwargs.pop('min_track_len')
        self.max_center_dist = kwargs.pop('max_center_dist')
        self.decay_time = kwargs.pop('decay_time')
        self.asso_thresh = kwargs.pop('asso_thresh')
        self.with_iou = kwargs.pop('with_iou')
        self.local_track = kwargs.pop('local_track')
        self.local_no_iou = kwargs.pop('local_no_iou')
        self.local_iou_only = kwargs.pop('local_iou_only')
        self.not_mult_thresh = kwargs.pop('not_mult_thresh')
        super().__init__(**kwargs)
        
        # self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
        self.noise_scheduler = DDIMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type = 'sample')
        self.net = ClassConditionedUnet().to(self.device)
        self.loss_fn = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3) 
        self.MatrixN = 80
        # self.fuck_max = 0
        # self.flag = False
        # self.Unet = Unet(
        #     dim = 64,
        #     dim_mults = (1, 2, 4, 8),
        #     num_classes = num_classes,
        #     cond_drop_prob = 0.5
        # )
        # self.diffusion = GaussianDiffusion(
        #     self.Unet,
        #     image_size = 512,
        #     timesteps = 1000,
        #     sampling_timesteps = 50
        # ).cuda()


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['test_len'] = cfg.INPUT.VIDEO.TEST_LEN
        ret['overlap_thresh'] = cfg.VIDEO_TEST.OVERLAP_THRESH     
        ret['asso_thresh'] = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        ret['min_track_len'] = cfg.VIDEO_TEST.MIN_TRACK_LEN
        ret['max_center_dist'] = cfg.VIDEO_TEST.MAX_CENTER_DIST
        ret['decay_time'] = cfg.VIDEO_TEST.DECAY_TIME
        ret['with_iou'] = cfg.VIDEO_TEST.WITH_IOU
        ret['local_track'] = cfg.VIDEO_TEST.LOCAL_TRACK
        ret['local_no_iou'] = cfg.VIDEO_TEST.LOCAL_NO_IOU
        ret['local_iou_only'] = cfg.VIDEO_TEST.LOCAL_IOU_ONLY
        ret['not_mult_thresh'] = cfg.VIDEO_TEST.NOT_MULT_THRESH
        return ret


    def remove_outdated(self,old_tensor, is_outdated) :
        new_tensor = torch.empty(1,1,self.MatrixN,0).to('cuda:0')
        for i in range( 1660 ) :
            if not is_outdated[i] :
                new_column = old_tensor[ : , : , : , i:i+1 ]
                new_tensor = torch.cat( (new_tensor,new_column) , dim = 3 )

        final_tensor = torch.full( (1,1,self.MatrixN,self.MatrixN) , -1.0 )
        _,_,_,m = new_tensor.shape
        assert m <= self.MatrixN
        final_tensor[ : , : , : , : m] = new_tensor
        return final_tensor

    def get_overlap_thresh( self , name ) :
        if 'MOT17-02' in name :
            return 0.1
        elif 'MOT17-04' in name :
            return 0.5
        elif 'MOT17-05' in name :
            return 0.1
        elif 'MOT17-09' in name :
            return 0.1
        elif 'MOT17-10' in name :
            return -0.7
        elif 'MOT17-11' in name :
            return 0.1
        elif 'MOT17-13' in name :
            return -0.2
        else :
            return 0.2


    def forward(self, batched_inputs):
        """
        All batched images are from the same video
        During testing, the current implementation requires all frames 
            in a video are loaded.
        TODO (Xingyi): one-the-fly testing
        """
        
        Fuck_point = False
        # Fuck_Min = 2.0
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs] # use ground-truth to test
        self.overlap_thresh = self.get_overlap_thresh( batched_inputs[0]['file_name'] )
        # import ipdb; ipdb.set_trace()
        if not self.training :
            import time
            total_time = 0
            diff_time = 0

            # import ipdb; ipdb.set_trace()
            self.noise_scheduler.set_timesteps( 25 , 'cuda:0' )
            N_list = torch.tensor([ len(x) for x in gt_instances ]) 
            N_presum_list = torch.cumsum( N_list , dim=0 )
            instance_list = []
            for frame_id in range(0,len(batched_inputs)) :
                if frame_id % 40 == 0 :
                    print(f'{frame_id} / {len(batched_inputs)}')
                    print(f'total time: {total_time}')
                    print(f'diff time: {diff_time}')

                total_st = time.time()


                this_instance = Instances( ( batched_inputs[0]['height'] , batched_inputs[0]['width'] ) )
                this_instance.set( 'scores' , torch.full( ( len(gt_instances[frame_id]) , ) , 1 ).to('cuda:0') )
                this_instance.set( 'pred_classes' , torch.full( ( len(gt_instances[frame_id]) , ) , 0).to('cuda:0') )
                this_instance.set('pred_boxes', 
                    box_post_precess( gt_instances[frame_id].gt_boxes ,
                                      (batched_inputs[0]['image'].shape[1],batched_inputs[0]['image'].shape[2]) , 
                                      (batched_inputs[0]['height'],batched_inputs[0]['width']) 
                                    ) )
                # instance format according to gtr structure

                if frame_id == 0 :
                    track_ids = torch.arange( 1 , len( gt_instances[0] ) + 1 )
                    this_instance.set( 'track_ids' , torch.arange( 1 , len( gt_instances[0] ) + 1 ) )
                    id_count = len( gt_instances[0] ) + 1
                else :


                    gt_instances_windows = gt_instances[max( frame_id-self.test_len+1 , 0 ):frame_id+1]                  # [id-31,id]
                    matching_matrix , Mat_Tn = get_matching_matrix(gt_instances_windows,self.MatrixN,1664)        # [1,1,n,Tn] Truth matching matrix, just for check
                    condition_matrix = get_distance_IOU_matrix(gt_instances_windows,self.MatrixN,1664) - 1        # [1,1,n,Tn] Iou condition matrix
                    
                    # decrease_matrix = get_decrease_matrix(gt_instances_windows)
                    # condition_decrease = make_matrix_decrease( condition_matrix , decrease_matrix ) - 1
                    
                    # if frame_id % 60 == 0 or Fuck_point :
                        # import ipdb; ipdb.set_trace()
                    is_outdated , id_map = get_outdated( instance_list[ max( frame_id-self.test_len+1 , 0 ) : ] , self.training , max( frame_id-self.test_len+1 , 0 ) )  # check whether each column is outdated 
                    condition_final = self.remove_outdated( condition_matrix , is_outdated ).to(self.device)       # condition matrix after removing outdating column
                    matching_final = self.remove_outdated( matching_matrix , is_outdated ).to(self.device)         # matching matrix


                    # for row in range( len( gt_instances[-1] ) ) :
                    #     one_bool = matching_final[0][0][row].eq(1)
                    #     one_indices = torch.nonzero( one_bool ).squeeze()
                    #     if one_indices.dim() == 1 :
                    #         continue
                    #     one_indices = one_indices.item()
                    #     This_value = condition_final[0][0][row][one_indices]
                    #     if This_value < Fuck_Min :
                    #         import ipdb; ipdb.set_trace()
                    #         Fuck_Min = This_value
                        # Fuck_Min = min( Fuck_Min , This_value )

                    #### Modify : Consider the decreasing coficient into the condition matrix

                    matching_generated = torch.randn(1,1,self.MatrixN,self.MatrixN).to(self.device)

                    diff_st = time.time()

                    for j, t in tqdm(enumerate(self.noise_scheduler.timesteps)):

                        with torch.no_grad():
                            residual = self.net(matching_generated, t, condition_final) 

                        # Update sample with step
                        matching_generated = self.noise_scheduler.step(residual, t, matching_generated).prev_sample    # remove the noise

                    diff_time += time.time() - diff_st
  
                    matching_sub = matching_generated[ : , : , 0:len( gt_instances[frame_id] ) , 0:len(id_map) ]       # just use the valid index
  
                    match_i , match_j = linear_sum_assignment( -matching_sub[0][0].cpu() )
                    track_ids = torch.full( (len( gt_instances[frame_id] ),) , -1 )
  
  
                    for i , j in zip( match_i , match_j ) :
                        # if matching_sub[0][0][i][j] > self.overlap_thresh :
                        if condition_final[0][0][i][j] > self.overlap_thresh :                                  # enough score to match
                        # if condition_decrease[0][0][i][j] > self.overlap_thresh * decrease_matrix[j] :
                            (id1,id2) = id_map[j]
                            if id1 < frame_id and i < len( gt_instances[frame_id] ) :
                                track_ids[i] = instance_list[id1]['instances'][id2].track_ids.item()            # get the id
                    
                    for i in range( len( gt_instances[frame_id] ) ) :
                        if track_ids[i] < 0 :
                            track_ids[i] = id_count
                            id_count += 1  
                                 
                    # if frame_id % 60 == 0 or Fuck_point :
                        # import ipdb; ipdb.set_trace()
                    # if frame_id >= 10 :
                    # import ipdb; ipdb.set_trace()

                    # if fuck_check( track_ids ) :
                    #     import ipdb; ipdb.set_trace()
                    
                    this_instance.set( 'track_ids' , track_ids.to('cuda:0') )
                    
                # import ipdb; ipdb.set_trace()
                boxxxx = []
                for i in range( len(gt_instances[frame_id]) ) :
                    boxxxx.append( gt_instances[frame_id].get('gt_boxes').tensor[i].tolist() )
                # import ipdb; ipdb.set_trace()
                # test = plot_tracking( batched_inputs[frame_id]['image'].permute(1,2,0) , boxxxx , gt_instances[frame_id].get('gt_instance_ids').tolist() , None , frame_id )
                test = plot_tracking( batched_inputs[frame_id]['image'].permute(1,2,0) , boxxxx , track_ids.tolist() , None , frame_id )
                filename = "./gtr/modeling/debug_ycy/img{}.png".format(frame_id)
                cv2.imwrite( filename , test )
                
                instance_list.append( {'instances':this_instance} )

                total_time += time.time() - total_st

            # print(Fuck_Min)
            return instance_list

        
        else :
            # import ipdb; ipdb.set_trace()
                        
            condition_matrix = get_distance_IOU_matrix(gt_instances,self.MatrixN,1664) - 1
            # decrease_matrix = get_decrease_matrix(gt_instances)
            # condition_decrease = make_matrix_decrease( condition_matrix , decrease_matrix ) - 1

            is_outdated , id_map = get_outdated( gt_instances[:-1],self.training , 0 )
            condition_final = self.remove_outdated( condition_matrix , is_outdated ).to(self.device)
            
    
            matching_matrix , _ = get_matching_matrix(gt_instances,self.MatrixN,1664)
            matching_final = self.remove_outdated( matching_matrix , is_outdated ).to(self.device)
            
            # if not correct_train( condition_final , matching_final ) : 
            #     import ipdb; ipdb.set_trace()
            # for row in range( len( gt_instances[-1] ) ) :
            #     one_bool = matching_final[0][0][row].eq(1)
            #     one_indices = torch.nonzero( one_bool ).squeeze()
            #     if one_indices.dim() == 1 :
            #         continue
            #     one_indices = one_indices.item()
            #     (id1,id2) = id_map[ one_indices ]
            #     if gt_instances[id1].get('gt_instance_ids')[id2].item() != gt_instances[-1].get('gt_instance_ids')[row].item() :
            # import ipdb; ipdb.set_trace()

            # check the correctness

            noise = torch.randn_like(matching_final).to(self.device)
            timesteps = torch.randint(0, 999, (matching_final.shape[0],)).long().to(self.device)
            noisy_x = self.noise_scheduler.add_noise( matching_final , noise , timesteps )
            # import ipdb; ipdb.set_trace()
            pred = self.net( noisy_x , timesteps , condition_final )
            
            loss = self.loss_fn( pred , matching_final )

            losses = {}
            losses['loss_diff'] = loss

            return losses


    def sliding_inference(self, batched_inputs):
        video_len = len(batched_inputs)
        instances = []
        id_count = 0
        for frame_id in range(video_len):
            if( frame_id % 20 == 0 ) :
                print(f"pencentage:{frame_id/video_len}")
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1],
                cfg,
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])

            if frame_id == 0: # first frame
                instances[0].track_ids = torch.arange(
                    1, len(instances[0]) + 1,
                    device=instances[0].reid_features.device)
                id_count = len(instances[0]) + 1
            else:
                win_st = max(0, frame_id + 1 - self.test_len)
                win_ed = frame_id + 1
                instances[win_st: win_ed], id_count = self.run_global_tracker(
                    batched_inputs[win_st: win_ed],
                    instances[win_st: win_ed],
                    k=min(self.test_len - 1, frame_id),
                    id_count=id_count) # n_k x N
            if frame_id - self.test_len >= 0:
                instances[frame_id - self.test_len].remove(
                    'reid_features')

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances


    def run_global_tracker(self, batched_inputs, instances, k, id_count):
        # import ipdb; ipdb.set_trace()
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)

        reid_features = torch.cat(
                [x.reid_features for x in instances], dim=0)[None]
        asso_output, pred_boxes, _, _ = self.roi_heads._forward_transformer(
            instances, reid_features, k) # [n_k x N], N x 4

        asso_output = asso_output[-1].split(n_t, dim=1) # T x [n_k x n_t]
        asso_output = self.roi_heads._activate_asso(asso_output) # T x [n_k x n_t]
        asso_output = torch.cat(asso_output, dim=1) # n_k x N

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np
        k_inds = [x for x in range(sum(n_t[:k]), sum(n_t[:k + 1]))]
        nonk_inds = [i for i in range(N) if not i in k_inds]
        asso_nonk = asso_output[:, nonk_inds] # n_k x Np
        k_boxes = pred_boxes[k_inds] # n_k x 4
        nonk_boxes = pred_boxes[nonk_inds] # Np x 4
        
        if self.roi_heads.delay_cls:
            # filter based on classification score similarity
            cls_scores = torch.cat(
                [x.cls_scores for x in instances], dim=0)[:, :-1] # N x (C + 1)
            cls_scores_k = cls_scores[k_inds] # n_k x (C + 1)
            cls_scores_nonk = cls_scores[nonk_inds] # Np x (C + 1)
            cls_similarity = torch.mm(
                cls_scores_k, cls_scores_nonk.permute(1, 0)) # n_k x Np
            asso_nonk[cls_similarity < 0.01] = 0

        unique_ids = torch.unique(ids) # M
        M = len(unique_ids) # number of existing tracks
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        # (n_k x Np) x (Np x M) --> n_k x M
        if self.decay_time > 0:
            # (n_k x Np) x (Np x M) --> n_k x M
            dts = torch.cat([x.reid_features.new_full((len(x),), T - t - 2) \
                for t, x in enumerate(instances) if t != k], dim=0) # Np
            asso_nonk = asso_nonk * (self.decay_time ** dts[None, :])

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M
        if id_inds.numel() > 0:
            last_inds = (id_inds * torch.arange(
                Np, device=id_inds.device)[:, None]).max(dim=0)[1] # M
            last_boxes = nonk_boxes[last_inds] # M x 4
            last_ious = pairwise_iou(
                Boxes(k_boxes), Boxes(last_boxes)) # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)
        
        if self.with_iou:
            traj_score = torch.max(traj_score, last_ious)
        
        if self.max_center_dist > 0.: # filter out too far-away trjactories
            # traj_score n_k x M
            k_boxes = pred_boxes[k_inds] # n_k x 4
            nonk_boxes = pred_boxes[nonk_inds] # Np x 4
            k_ct = (k_boxes[:, :2] + k_boxes[:, 2:]) / 2
            k_s = ((k_boxes[:, 2:] - k_boxes[:, :2]) ** 2).sum(dim=1) # n_k
            nonk_ct = (nonk_boxes[:, :2] + nonk_boxes[:, 2:]) / 2
            dist = ((k_ct[:, None] - nonk_ct[None, :]) ** 2).sum(dim=2) # n_k x Np
            norm_dist = dist / (k_s[:, None] + 1e-8) # n_k x Np
            # id_inds # Np x M
            valid = norm_dist < self.max_center_dist # n_k x Np
            valid_assn = torch.mm(
                valid.float(), id_inds).clamp_(max=1.).long().bool() # n_k x M
            traj_score[~valid_assn] = 0 # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.overlap_thresh * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.overlap_thresh
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        for i in range(n_k):
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
        instances[k].track_ids = track_ids

        assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        return instances, id_count


    def _remove_short_track(self, instances):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        num_insts_track = id_inds.sum(dim=1) # M
        remove_track_id = num_insts_track < self.min_track_len # M
        unique_ids[remove_track_id] = -1
        ids = unique_ids[torch.where(id_inds.permute(1, 0))[1]]
        ids = ids.split([len(x) for x in instances])
        for k in range(len(instances)):
            instances[k] = instances[k][ids[k] >= 0]
        return instances


    def _delay_cls(self, instances, video_id):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        M = len(unique_ids) # #existing tracks
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        # update scores
        cls_scores = torch.cat(
            [x.cls_scores for x in instances], dim=0) # N x (C + 1)
        traj_scores = torch.mm(id_inds, cls_scores) / \
            (id_inds.sum(dim=1)[:, None] + 1e-8) # M x (C + 1)
        _, traj_inds = torch.where(id_inds.permute(1, 0)) # N
        cls_scores = traj_scores[traj_inds] # N x (C + 1)

        n_t = [len(x) for x in instances]
        boxes = [x.pred_boxes.tensor for x in instances]
        track_ids = ids.split(n_t, dim=0)
        cls_scores = cls_scores.split(n_t, dim=0)
        instances, _ = custom_fast_rcnn_inference(
            boxes, cls_scores, track_ids, [None for _ in n_t],
            [x.image_size for x in instances],
            self.roi_heads.box_predictor[-1].test_score_thresh,
            self.roi_heads.box_predictor[-1].test_nms_thresh,
            self.roi_heads.box_predictor[-1].test_topk_per_image,
            self.not_clamp_box,
        )
        for inst in instances:
            inst.track_ids = inst.track_ids + inst.pred_classes * 10000 + \
                video_id * 100000000
        return instances

    def local_tracker_inference(self, batched_inputs):
        from ...tracking.local_tracker.fairmot import FairMOT
        local_tracker = FairMOT(
            no_iou=self.local_no_iou,
            iou_only=self.local_iou_only)

        video_len = len(batched_inputs)
        instances = []
        ret_instances = []
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])
            inst = instances[frame_id]
            dets = torch.cat([
                inst.pred_boxes.tensor, 
                inst.scores[:, None]], dim=1).cpu()
            id_feature = inst.reid_features.cpu()
            tracks = local_tracker.update(dets, id_feature)
            track_inds = [x.ind for x in tracks]
            ret_inst = inst[track_inds]
            track_ids = [x.track_id for x in tracks]
            ret_inst.track_ids = ret_inst.pred_classes.new_tensor(track_ids)
            ret_instances.append(ret_inst)
        instances = ret_instances

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances
