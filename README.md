# DeepRock
Rock Music using Deep Learning. 
For more information regarding the architecture and model go to https://nayak.io/posts/deeprock-rock-music-generation-using-lstm/

## Usage

1. Generate the required `X` and `Y` for training by running `python create_data.py`. You can mess around in __create_data.py__ to change the _sequence_length_ and _unique_factor_ parameters.
2. Modify the model as per your requirements in __train.py__. To train run `python train.py`.
3. Generate new music from random samples by running `python generate.py`.

## Credits
1. https://github.com/jisungk/deepjazz
2. https://github.com/Skuldur/Classical-Piano-Composer
