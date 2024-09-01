# Model Scaling Behavior Analysis

This project analyzes the scaling behavior of different training approaches for language models.

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- Matplotlib
- Pandas
- Scikit-learn
- OpenAI (for API access)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/model-scaling-behavior.git
   cd model-scaling-behavior
   ```

2. Install the required packages:
   ```
   pip install torch transformers matplotlib pandas scikit-learn openai
   ```

3. Set up your OpenAI API key:
   - Create a file named `.env` in the project root directory
   - Add your API key to the file:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

## Usage

1. Ensure you're in the project directory:
   ```
   cd model-scaling-behavior
   ```

2. Run the main script to execute the experiments and generate visualizations:
   ```
   python main.py
   ```

   This will:
   - Load and preprocess the data
   - Run baseline, auxiliary confidence loss, bootstrapping, and generative fine-tuning experiments
   - Generate plots for scaling behavior and error distribution
   - Print the results of each experiment

3. After running the script, you'll find two new image files in the project directory:
   - `scaling_behavior.png`: A plot showing the scaling behavior of different training approaches
   - `error_distribution.png`: A histogram of the error distribution for the strong model

4. The console will display the results of each experiment, showing the accuracy for small, medium, and large models.

## Customization

- To use your own dataset, modify the `load_and_preprocess_data()` function in `data_utils.py`.
- To adjust the model architectures or fine-tuning process, edit the functions in `models.py`.
- To change the experimental procedures, update the functions in `experiments.py`.
- To modify the visualizations, edit the functions in `visualization.py`.

## File Structure

- `main.py`: Main script to run the experiments
- `data_utils.py`: Functions for loading and preprocessing data
- `models.py`: Functions for loading and fine-tuning models
- `experiments.py`: Implementation of different experimental approaches
- `visualization.py`: Functions for plotting results
- `colab_runner.ipynb`: Jupyter notebook for running experiments in Google Colab

## Notes

- This implementation uses simplified placeholder functions for demonstration purposes.
- You may need to modify the code to work with your specific datasets and requirements.
- Ensure you have sufficient computational resources when working with large language models.

## Troubleshooting

If you encounter any issues:
1. Make sure all required packages are installed correctly.
2. Check that you're using a compatible version of Python (3.7+).
3. Verify that you have sufficient disk space for model downloads and generated plots.
4. If you're using a GPU, ensure that PyTorch is configured correctly for GPU usage.
5. Make sure your OpenAI API key is set up correctly in the `.env` file.

For any persistent problems, please open an issue on the GitHub repository.
