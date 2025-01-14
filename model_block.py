import importlib
raft_stereo = importlib.import_module("thirdparty.RAFT-Stereo.core.raft_stereo")
utils = importlib.import_module("thirdparty.RAFT-Stereo.core.utils.utils")

#from raft_stereo import RAFTStereo
#from utils.utils import InputPadder


import numpy as np
import torch
import cv2
import gc

class FastRAFTStereoParams:
    def __init__(self) -> None:
        self.hidden_dims = [128]*3
        self.n_downsample = 3
        self.shared_backbone = True
        self.mixed_precision = True
        self.corr_implementation = "reg"
        self.corr_radius = 4
        self.corr_levels = 4
        self.n_gru_layers = 2
        self.slow_fast_gru = True
        self.valid_iters = 7

class RAFTStereoParams:
    def __init__(self) -> None:
        self.hidden_dims = [128]*3
        self.n_downsample = 2
        self.shared_backbone = False
        self.mixed_precision = False
        self.corr_implementation = "reg"
        self.corr_radius = 4
        self.corr_levels = 4
        self.n_gru_layers = 3
        self.slow_fast_gru = False
        self.valid_iters = 32

class RAFTStereoBLock:
    def __init__(self, device = "cpu", use_fast = True, verbose=False): 
        
        self.logName = "RAFT-Stereo Block"
        self.verbose = verbose
        
        self.device = device
        self.use_fast = use_fast
        self.model_params = FastRAFTStereoParams() if use_fast else RAFTStereoParams()
        self.disposed = False

    def log(self, x):
        if self.verbose:
            print(f"{self.logName}: {x}")

    def build_model(self):
        if self.disposed:
            self.log("Session disposed!")
            return

        logline = "fast" if self.use_fast else ""
        self.log(f"Building Model RAFT-stereo {logline} ...")

        self.model = torch.nn.DataParallel(raft_stereo.RAFTStereo(self.model_params)).to(self.device)

    def load(self, model_path):
        self.log("Loading frozen model")
        self.log(f"Model checkpoint path: {model_path}")

        state_dict = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(state_dict["state_dict"] if "state_dict" in state_dict else state_dict)

        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()


    def dispose(self):
        if not self.disposed:
            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.disposed = True

    def _conv_image(self,imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]

        conv_list = []

        for img in imgs:
            if len(img.shape) < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = torch.from_numpy(img).permute(2, 0, 1).float()
            conv_list.append(img.unsqueeze(0).to(self.device))
        
        return torch.cat(conv_list, 0)

    def test(self, left, right, left_vpp, right_vpp, normalize=True):
        #Input conversion
        left = self._conv_image(left)
        right = self._conv_image(right)
        left_vpp = self._conv_image(left_vpp)
        right_vpp = self._conv_image(right_vpp)

        with torch.no_grad():
            padder = utils.InputPadder(left.shape, divis_by=32)
            left, right, left_vpp, right_vpp = padder.pad(left, right, left_vpp, right_vpp)

            _, flow_up = self.model(left, right, left_vpp, right_vpp, iters=self.model_params.valid_iters, test_mode=True, normalize=normalize)

            flow_pr = padder.unpad(flow_up.float()).cpu().squeeze(0)

            dmap = flow_pr.cpu().numpy().squeeze()
            dmap = -dmap

            return dmap
            
    def vpp_train(self, left, right, left_vpp, right_vpp):
        #Input conversion
        left = self._conv_image(left)
        right = self._conv_image(right)

        # Assume that a vpp network already do _conv_image
        # left_vpp = self._conv_image(left_vpp)
        # right_vpp = self._conv_image(right_vpp)

        # Freeze model, but enable backprop
        for param in self.model.parameters():
            param.requires_grad = False
        
        padder = utils.InputPadder(left.shape, divis_by=32)
        left, right, left_vpp, right_vpp = padder.pad(left, right, left_vpp, right_vpp)

        _, flow_up = self.model(left, right, left_vpp, right_vpp, iters=self.model_params.valid_iters, test_mode=True)

        flow_pr = padder.unpad(flow_up.float()).squeeze(0)

        dmap = flow_pr.squeeze()
        dmap = -dmap

        return dmap
            