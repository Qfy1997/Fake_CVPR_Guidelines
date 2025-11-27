"""
https://hf-mirror.com/openai/clip-vit-base-patch32
model card 
Use with Transformers 
"""

from PIL import Image

from transformers import CLIPProcessor, CLIPModel

if __name__=='__main__':
    #加载模型
    model = CLIPModel.from_pretrained("clip/")
    processor = CLIPProcessor.from_pretrained("clip/")
    figure_path = "000000039769.jpg" # 对应当前文件夹下本地图片的位置
    # 加载图片
    image = Image.open(figure_path)
    # 对text中的文本以及图片进行表示
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    # 输入到模型中
    outputs = model(**inputs)
    # ['logits_per_image', 'logits_per_text', 'text_embeds', 'image_embeds', 'text_model_output', 'vision_model_output']
    print(outputs.text_embeds.shape)
    print(outputs.image_embeds.shape)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    print(logits_per_image)
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    print(probs.shape)
