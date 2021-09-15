# ProtoSeq
python3 emotionClf.py \
    --task fsl_emoseq \
    --encoder cnnlstm_seq \
    --crf \
    --patience 2 \
    --train_epochs 10000 \
    --patience_metric f1_micro \
    --cuda 0

# WarmProto
python3 emotionClf.py \
    --task fsl_emoseq \
    --encoder rnn_seq \
    --crf \
    --warmproto \
    --patience 2 \
    --train_epochs 10000 \
    --patience_metric f1_micro \
    --cuda 0

# Proto
python3 emotionClf.py \
    --task fsl_emoseq \
    --encoder cnn_seq \
    --classifier proto_seq \
    --patience 2 \
    --train_epochs 10000 \
    --patience_metric f1_micro \
    --cuda 0

## Variants showed in the paper
# ProtoSeq-AVG
python3 emotionClf.py \
    --task fsl_emoseq \
    --encoder avg_seq \
    --classifier proto_seq \
    --crf \
    --patience 2 \
    --train_epochs 10000 \
    --patience_metric f1_micro \
    --cuda 0
    --tiny
# ProtoSeq-Tr
python3 emotionClf.py \
    --task fsl_emoseq \
    --encoder transfo_seq \
    --classifier proto_seq \
    --crf \
    --patience 2 \
    --train_epochs 10000 \
    --patience_metric f1_micro \
    --cuda 0
# ProtoSeq-CNN
python3 emotionClf.py \
    --task fsl_emoseq \
    --encoder cnn_seq \
    --classifier proto_seq \
    --crf \
    --patience 2 \
    --train_epochs 10000 \
    --patience_metric f1_micro \
    --cuda 0

# CESTa supervised baseline
python3 emotionClf.py \
    --task supervised_emoseq \
    --encoder cnn_seq \
    --classifier cesta \
    --patience 2 \
    --train_epochs 10000 \
    --patience_metric f1_micro \
    --cuda 0