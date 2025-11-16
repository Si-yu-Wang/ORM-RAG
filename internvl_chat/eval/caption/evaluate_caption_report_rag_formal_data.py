import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import sys
import datetime
from evaluate_acc import calculate_acc

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = cur_path[:cur_path.find('internvl_chat')] + 'internvl_chat'
print(root_path)
sys.path.append(root_path)

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoModel

ds_collections = {
    'rag_test_en_formal_retrieval_data_faiss_one_to_one': {
        'root': '/',
        'annotation': '/public/Report-Ge/code/InternVL-wsy/internvl_chat/datasets/datasets/coco_en_test_with_retrieval_info_one_to_one.json',
        'max_new_tokens': 1200,
        'min_new_tokens': 8,
    }
}

class CaptionDataset(torch.utils.data.Dataset):

    def __init__(self, name, root, annotation, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6, rag_usage=False):
        # if name == 'coco':
        #     self.images = json.load(open(annotation))
        # else:
        self.images = json.load(open(annotation))['images']
        # self.prompts = json.load(open(annotation))['images']
        # for i in range(len(self.images)):
        #     image_file = self.images[i]['file_name']
        #     if len(image_file)==0:
        #         print(self.images[i])
        # sss
        self.name = name
        self.prompt = prompt
        self.root = root
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)
        self.rag_usage=rag_usage

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # if self.name == 'coco':
        #     filename = self.images[idx]['image']
        #     image_id = int(filename.split('_')[-1].replace('.jpg', ''))
        #     image_path = os.path.join(self.root, filename)
        # else:
        image_id = self.images[idx]['id']
        image_file = self.images[idx]['file_name']
        images, num_tiles = [], []
        rag_images=[]
        image_tensors = []
        num_image = len(image_file)
        # history_images = []
        # history_reports = []
        #=================================================================================================================================
        if self.rag_usage:
            if 'retrieval_images' in self.images[idx] and len(self.images[idx]['retrieval_images']) > 0:
                relative_image=self.images[idx]['retrieval_images']
                relative_report=self.images[idx]['retrieval_reports']
                # print("relative_image:",relative_image)
                # print("relative_report:",relative_report)

                relative_images_num = len(relative_image)

                for rag_image_path in relative_image: #多维向量[[][]],这种情况
                    # Merge the image path
                    
                    # Load the image using tcs_loader if available, otherwise use PIL
                    try:
                        rag_image =  Image.open(rag_image_path).convert('RGB')
                    except Exception as e:
                        print(f"加载图片 {rag_image_path} 失败: {e}")
                    if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                        rag_images = dynamic_preprocess(rag_image, min_num=self.min_dynamic_patch,
                                                max_num=self.max_dynamic_patch // num_image,
                                                image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                        rag_images += rag_image
                        # num_tiles.append(len(rag_image))
                    else:  # Otherwise, use the original image as a single patch
                        rag_images.append(rag_image)
                        # num_tiles.append(1)
                # assert len(rag_images) > 0, f'image file is zero, {image_id},{rag_images}\npredicted_label:{predicted_labels}\nrelative_image:{relative_image}\njudge_labels:{judge_labels},\nimage_file:{image_file}\n'
                
                rag_pixel_values = [self.transform(rag_image) for rag_image in rag_images]
                rag_pixel_values = torch.stack(rag_pixel_values)
            else:
                relative_images_num=0
                rag_pixel_values=None
                relative_report=[]

        # =================================================================================================================================================

        for image_path in image_file:
            # Merge the image path
            
            # Load the image using tcs_loader if available, otherwise use PIL
            try:
                image =  Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"加载图片 {rag_image_path} 失败: {e}")
            if self.dynamic_image_size:  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(image, min_num=self.min_dynamic_patch,
                                           max_num=self.max_dynamic_patch // num_image,
                                           image_size=self.image_size, use_thumbnail=self.use_thumbnail)
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        if len(images) == 0:
            assert f'image file is zero, {image_id}'
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        # print("pixel_values:", pixel_values.shape)

        # image = Image.open(image_path)
        # if self.dynamic_image_size:
        #     images = dynamic_preprocess(image, image_size=self.input_size,
        #                                 use_thumbnail=self.use_thumbnail,
        #                                 max_num=self.max_num)
        # else:
        #     images = [image]
        # pixel_values = [self.transform(image) for image in images]
        # pixel_values = torch.stack(pixel_values)
        if self.rag_usage:
            return {
                'image_id': image_id,
                'input_text': self.prompt,
                'pixel_values': pixel_values,
                'rag_pixel_values': rag_pixel_values,
                'relative_images_num':relative_images_num,
                'relative_report':relative_report
            }
        else:
            return {
                'image_id': image_id,
                'input_text': self.prompt,
                'pixel_values': pixel_values
            }

def collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')

    return pixel_values, image_ids, input_tokens.input_ids, input_tokens.attention_mask

def rag_collate_fn(inputs, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in inputs], dim=0)
    image_ids = [_['image_id'] for _ in inputs]
    input_texts = [_['input_text'] for _ in inputs]
    input_tokens = tokenizer(input_texts, return_tensors='pt')
    if inputs[0]['rag_pixel_values'] is not None:
        rag_pixel_values = torch.cat([sample['rag_pixel_values'] for sample in inputs], dim=0)
        relative_images_num = [sample['relative_images_num'] for sample in inputs]
        relative_report = [sample['relative_report'] for sample in inputs]
    else:
        rag_pixel_values = None
        relative_images_num = None
        relative_report = None
    # print("rag_pixel_values:", rag_pixel_values.shape)

    return (pixel_values, image_ids, input_tokens.input_ids,
            input_tokens.attention_mask, rag_pixel_values,
            relative_images_num, relative_report)

    # return pixel_values, image_ids, input_tokens.input_ids, input_tokens.attention_mask


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size, self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def evaluate_chat_model():
    prompt = 'Please generate a fetal ultrasound examination report according to the similar images and reports retrieved above and the multiple images I gave. The retrieval information may contain the same abnormal parts as the multiple images I gave, or it may be different. Please make a specific judgment according to the multiple images I gave.'
    print('prompt:', prompt)
    random.seed(args.seed)
    summaries = []

    for ds_name in args.datasets:
        annotation = ds_collections[ds_name]['annotation']
        if type(annotation) == list:
            annotation = annotation[0]
        dataset = CaptionDataset(
            name=ds_name,
            root=ds_collections[ds_name]['root'],
            annotation=annotation,
            prompt=prompt,
            input_size=image_size,
            dynamic_image_size=args.dynamic,
            use_thumbnail=use_thumbnail,
            max_num=args.max_num,
            rag_usage=args.rag_usage
        )
        if args.rag_usage:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=InferenceSampler(len(dataset)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=partial(rag_collate_fn, tokenizer=tokenizer),
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                sampler=InferenceSampler(len(dataset)),
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                collate_fn=partial(collate_fn, tokenizer=tokenizer),
            )

        image_ids, captions = [], []
        if args.rag_usage:
            for _, (pixel_values, ids, _, _, rag_pixel_values, num_ref_image, reference_reports) in tqdm(enumerate(dataloader)):
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                if rag_pixel_values is not None:
                    rag_pixel_values = rag_pixel_values.to(torch.bfloat16).cuda()
                generation_config = dict(
                    num_beams=args.num_beams,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                )
                with torch.no_grad():
                    pred = model.rag_chat( 
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=prompt,
                        generation_config=generation_config,
                        reference_reports=reference_reports,
                        rag_pixel_values=rag_pixel_values,
                        num_ref_image=num_ref_image,
                        verbose=False
                    )
                    image_ids.extend(ids)
                    captions.extend([pred])
                    torch.cuda.empty_cache()
        else:
            for _, (pixel_values, ids, _, _) in tqdm(enumerate(dataloader)):
                pixel_values = pixel_values.to(torch.bfloat16).cuda()
                generation_config = dict(
                    num_beams=args.num_beams,
                    max_new_tokens=ds_collections[ds_name]['max_new_tokens'],
                    min_new_tokens=ds_collections[ds_name]['min_new_tokens'],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                )
                with torch.no_grad():
                    pred = model.chat( 
                        tokenizer=tokenizer,
                        pixel_values=pixel_values,
                        question=prompt,
                        generation_config=generation_config,
                        verbose=False
                    )
                    image_ids.extend(ids)
                    captions.extend([pred])
                    torch.cuda.empty_cache()

        torch.distributed.barrier()

        world_size = torch.distributed.get_world_size()
        merged_ids = [None for _ in range(world_size)]
        merged_captions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_ids, image_ids)
        torch.distributed.all_gather_object(merged_captions, captions)

        merged_ids = [_ for _ in itertools.chain.from_iterable(merged_ids)]
        merged_captions = [_ for _ in itertools.chain.from_iterable(merged_captions)]
        average_length = sum(len(x.split()) for x in merged_captions) / len(merged_captions)
        print(f'Average caption length: {average_length}')

        if torch.distributed.get_rank() == 0:
            print(f'Evaluating {ds_name} ...')

            results = []
            for image_id, caption in zip(merged_ids, merged_captions):
                results.append({
                    'image_id': int(image_id),
                    'caption': caption,
                })
            time_prefix = time.strftime('%y%m%d%H%M%S', time.localtime())
            results_file = f'{ds_name}_{time_prefix}.json'
            results_file = os.path.join(args.out_dir, results_file)
            json.dump(results, open(results_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

            annotation = ds_collections[ds_name]['annotation']
            acc_metric = calculate_acc(annotation, results_file)
            if type(annotation) == list:
                annotation = annotation[-1]
            coco = COCO(annotation)
            coco_result = coco.loadRes(results_file)
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.evaluate()

            summary = coco_eval.eval.items()
            print(summary)
            summaries.append([args.checkpoint, results_file, ds_name, average_length, summary])
            summaries.append(acc_metric)

        torch.distributed.barrier()

    out_path = '_'.join(args.checkpoint.split('/')[-2:])
    writer = open(os.path.join(args.out_dir, f'{out_path}.txt'), 'a')
    print(f"write results to file {os.path.join(args.out_dir, f'{out_path}.txt')}")
    for summary in summaries:
        print(summary)
        writer.write(f'{summary}\n')
    writer.close()

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--datasets', type=str, default='report')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--num-beams', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--out_dir', type=str, default='wsy_results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--rag_usage', action='store_true', help='是否启用 rag_usage')
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(',')
    print('datasets:', args.datasets)
    assert args.batch_size == 1, 'Only batch size 1 is supported'

    torch.distributed.init_process_group(
        backend='nccl',
        timeout=datetime.timedelta(minutes=300),
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )

    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')
    print(f'[test] rag_usage: {args.rag_usage}')

    if args.rag_usage:
        print("=====================================================启用rag=====================================================")
        model_name = "/public/Report-Ge/code/InternVL-wsy/internvl_chat/vit"  # 替换为实际路径
        rag_model = AutoModel.from_pretrained(model_name)
        # 设备设置
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model = model.to(device)
        # classifier = classifier.to(device)
        # 加载模型权重
        rag_checkpoint_path=f'/public/Report-Ge/code/InternVL-wsy/internvl_chat/test/16_labels_task/vit_epoch_13_checkpoint.pth'
        rag_checkpoint = torch.load(rag_checkpoint_path,map_location='cpu')
        rag_model.load_state_dict(rag_checkpoint['model_state_dict'])
        # 评估模式
        rag_model.eval()  # 将模型切换到评估模式
    else:
        print("=====================================================未启用rag=====================================================")

    evaluate_chat_model()
