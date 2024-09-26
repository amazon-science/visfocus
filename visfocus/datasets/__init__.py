DATASETS_DIR = '/mnt/efs/ofirab/datasets'


from .infovqa import infographicsVQADataset, InfoVQADataset_P2S
from .docvqa import DocVQADataset, DocVQADataset_P2S
from .chartqa import ChartQADataset, ChartQADataset_Eval_P2S # ChartQADataset_P2S
from .ai2d import AI2D
from .ocrvqa import OCRVQA

DATASETS = {
    'docvqa': DocVQADataset,
    'infovqa': infographicsVQADataset,
    'infovqa_p2s': InfoVQADataset_P2S,
    'chartqa_augmented': ChartQADataset,
    'chartqa_human': ChartQADataset,
    'docvqa_p2s_train': DocVQADataset_P2S,
    'ai2d': AI2D,
    'ocrvqa': OCRVQA,
    'chartqa_augmented_p2s': ChartQADataset_Eval_P2S,
    'chartqa_human_p2s': ChartQADataset_Eval_P2S,
}