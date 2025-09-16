from .sts_loss import STSLoss
from .dual_space_kd_with_cross_model_attention import DualSpaceKDWithCMA
from .rmse_cka import RMSE_CKA
from .ot_pro import OT_PRO
from .ot_pro_rmse_cka import OT_PRO_RMSE_CKA
from .FKD import FKD
from .FKD_A import FKD_A
from .FKD_H import FKD_H

criterion_list = {
    "sts_loss": STSLoss,
    "dual_space_kd_with_cross_model_attention": DualSpaceKDWithCMA,
    "rmse_cka": RMSE_CKA,
    "ot_pro": OT_PRO,
    "ot_pro_rmse_cka": OT_PRO_RMSE_CKA,
    "fkd": FKD,
    "fkd_a": FKD_A,
    "fkd_h": FKD_H
}

def build_criterion(args):
    if criterion_list.get(args.criterion, None) is not None:
        return criterion_list[args.criterion](args)
    else:
        raise NameError(f"Undefined criterion for {args.criterion}!")
