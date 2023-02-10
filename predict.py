from test import *

from cog import BasePredictor, Path, Input

class Predictor(BasePredictor):
    def setup(self):
        self.model_func = load_model('crestereo_eth3d.mge')

    def predict(
            self,
            left_image: Path = Input(description="Path to an image"),
            right_image: Path = Input(description="Path to an image"),
            # inference_height: int = Input(default=1024,  description="Model name"),
            # inference_width: int = Input(default=1536,  description="Model name"),
    ) -> Path:
        output_path = "output.png"
        left = cv2.imread(str(left_image))
        right = cv2.imread(str(right_image))

        assert left.shape == right.shape, "The input images have inconsistent shapes."

        in_h, in_w = left.shape[:2]

        eval_h, eval_w = 1024,1536
        left_img = cv2.resize(left, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        right_img = cv2.resize(right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

        pred = inference(left_img, right_img, self.model_func, n_iter=20)

        t = float(in_w) / float(eval_w)
        disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

        disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
        disp_vis = disp_vis.astype("uint8")
        disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

        cv2.imwrite(output_path, disp_vis)

        return Path(output_path)