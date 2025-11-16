GPUS=3
MASTER_PORT=1234
CUDA_VISIBLE_DEVICES="0, 1, 2" torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --nproc_per_node=${GPUS} \
    --master_port=${MASTER_PORT} \
    eval/caption/evaluate_caption_report_rag_formal_data.py --rag_usage --checkpoint /public/Report-Ge/code/InternVL-wsy/internvl_chat/wsy_output/final/formal_retrieval_data_faiss_highhighlevel_BS_2_epoch_3 --datasets rag_test_en_formal_retrieval_data_faiss_one_to_one