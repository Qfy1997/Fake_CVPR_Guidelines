需要到官网下载对应的模型文件，国内一般用huggingface mirror访问。
https://hf-mirror.com/openai/clip-vit-base-patch32
所需文件：
config.json
merges.txt
preprocessor_config.json
pytorch_model.bin
special_tokens_map.json
tokenizer_config.json
tokenizer.json
vocab.json
将所需要的文件下载到本地并整理到一个文件夹中。

环境中需要加载的包：
python
transformer
PIL
