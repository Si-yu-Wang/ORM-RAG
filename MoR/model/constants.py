BASE_OUTPUT = "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/test"
BASE_TRAIN_DATA = "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/train_Data"
BASE_TRAIN_DATA = "/public/Report-Ge/code/InternVL-wsy/internvl_chat/translate"
SKIPPED_GROUPS = {1, 7}  # do not build indices for these label groups
# label_id 对应的组名
_GROUP_NAMES = {
    0: "腹部_group_0",
    1: "脊柱_group_1",
    2: "头部_group_2",
    3: "四肢_group_3",
    4: "胎盘_group_4",
    5: "心脏_group_5",
    6: "面部_group_6",
    7: "肾脏_group_7",
}
CKPT_DICT = {
    0: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/腹部_best_model.pth",
    1: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/脊柱_best_model.pth",
    2: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/头部_best_model.pth",
    3: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/四肢_best_model.pth",
    4: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/胎盘_best_model.pth",
    5: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/心脏_best_model.pth",
    6: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/面部_best_model.pth",
    7: "/public/Report-Ge/code/InternVL-wsy/internvl_chat/build_retrieval_faiss/final_model/肾脏_best_model.pth"
}
VIT_MEAN = [0.5, 0.5, 0.5]
VIT_STD = [0.5, 0.5, 0.5]
IMAGENET_MEAN=[0.485, 0.456, 0.406]
IMAGENET_STD=[0.229, 0.224, 0.225]

# _GROUP_NAMES = {
#     0: "腹部_group_0_updated",
#     1: "脊柱_group_1_updated",
#     2: "头部_group_2_updated",
#     3: "四肢_group_3_updated",
#     4: "胎盘_group_4_updated",
#     5: "心脏_group_5_updated",
#     6: "面部_group_6_updated",
#     7: "肾脏_group_7_updated"
# }