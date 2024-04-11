# Project Title: Mixtral Model Expert Extractor

This project is designed to extract all the experts from a Mixtral model, which is a mixture of experts model. It forms `n` Mistral models where `n` is the number of experts in the Mixtral model.

## Prerequisites

The project is developed using Python and requires the following libraries:

- transformers
- torch
- argparse
- os


## Usage

The main script of the project is `main.py`. It takes two command-line arguments:

- `--model_id`: The path to the pretrained Mixtral model.
- `--target_dir`: The directory where the extracted Mistral models will be saved.

Here is an example of how to run the script:

```bash
python main.py --model_id "/workspace/text-generation-webui2/models/mistralai_Mixtral-8x7B-Instruct-v0.1" --target_dir "./experts"
```

This will extract all the experts from the Mixtral model and save them as individual Mistral models in the `./experts` directory.

## Walkthrough Example

Because Mixtral models could be large, to test the script, we have created a `create_dummy_mixtral_for_test.py` file to create a small dummy Mixtral model. This model is saved to `./mixtral-makeup`. You can run this script and then try to extract experts from it using the following commands:

```bash
python create_dummy_mixtral_for_test.py
python main.py --model_id ./mixtral-makeup --target_dir ./experts
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Extracted Models

The 8 models extracted from Mixtral-8x7B are now available in the Hugging Face Model Hub. You can find them in the collection [here](https://huggingface.co/collections/DrNicefellow/extracted-models-from-mixtral-8x7b-6617d7cc0da4c017c4d15c96).

## Limitations

Currently, the script extracts the same index of experts at each block/layer. However, it's important to note that the indices at different blocks/layers are irrelevant. This means that the expert with the same index in different layers does not necessarily correspond to the same expert. This is a limitation of the current implementation and users should be aware of this when using the extracted models.

## Disclaimer

This software is provided "as is", without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software.

Users are responsible for checking and validating the correctness of their configuration files, safetensor files, and binary files generated using the software. The developers assume no responsibility for any errors, omissions, or other issues coming in these files, or any consequences resulting from the use of these files.


## License

[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Discord Server

Join our Discord server [here](https://discord.gg/xhcBDEM3).

## Feeling Generous? 😊

Eager to buy me a cup of 2$ coffe or iced tea?🍵☕ Sure, here is the link: [https://ko-fi.com/drnicefellow](https://ko-fi.com/drnicefellow). Please add a note on which one you want me to drink?