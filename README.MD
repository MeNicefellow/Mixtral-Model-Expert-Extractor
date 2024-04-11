# README.MD

## Project Title: Mixtral Model Expert Extractor

This project is designed to extract all the experts from a Mixtral model, which is a mixture of experts model. It forms `n` Mistral models where `n` is the number of experts in the Mixtral model.

### Prerequisites

The project is developed using Python and requires the following libraries:

- transformers
- torch
- argparse
- os


### Usage

The main script of the project is `main.py`. It takes two command-line arguments:

- `--model_id`: The path to the pretrained Mixtral model.
- `--target_dir`: The directory where the extracted Mistral models will be saved.

Here is an example of how to run the script:

```bash
python main.py --model_id "/workspace/text-generation-webui2/models/mistralai_Mixtral-8x7B-Instruct-v0.1" --target_dir "./experts"
```

This will extract all the experts from the Mixtral model and save them as individual Mistral models in the `./experts` directory.

### Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

### License

[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Discord Server

Join our Discord server [here](https://discord.gg/xhcBDEM3).

## Feeling Generous? 😊

Eager to buy me a cup of 2$ coffe or iced tea?🍵☕ Sure, here is the link: [https://ko-fi.com/drnicefellow](https://ko-fi.com/drnicefellow). Please add a note on which one you want me to drink?