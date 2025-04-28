# QTM-347 Project

## Career Path Analysis and Prediction

This project uses machine learning techniques to analyze and predict career trajectories based on job history data.

## Goal of this Project

We want to provide a machine learning solution that helps identify the most likely next occupation or career move based on past experiences.

## Application & Motivation

- Why Transformer? A Transformer can model the full sequence of past jobs to predict the next role more accurately
- Better predictions could inform career-planning tools, HR systems, and personalized learning recommendations

## Pipeline

![Project Pipeline](docs/pipeline.png)

## Project Structure

- `code/career_data_utils.py`: Utility functions for processing career data
- `code/Karrierewege_plus_transformer_v2.py`: Transformer-based model implementation
- `code/lstm_code_v3.ipynb`: LSTM model implementation notebook
- `models/best_career_transformer_model_clustered.pth`: Trained transformer model checkpoint
- `results/cluster_assignments.txt`: Results from job clustering analysis

## Getting Started

### Setup
1. Clone this repository
2. Install required packages using pip:
   ```
   pip install -r requirements.txt
   ```
3. Run the models using provided scripts or notebooks:
   ```
   # To run the transformer model
   python code/Karrierewege_plus_transformer_v2.py
   
   # To use the LSTM model, open the notebook
   jupyter notebook code/lstm_code_v3.ipynb
   ```

## Models

The project implements and compares two main approaches:
1. **Transformer-based model**: For sequence modeling of career paths
2. **LSTM model**: For sequential prediction of career transitions

## Results

### Transformer Model Performance
- Lowest validation loss (tuning): 3.0444
- Test loss: 3.0047
- Test accuracy: 0.2284

### Best Hyperparameters
- Transformer layers: 1
- Attention heads: 2
- Batch size: 32
- Learning rate: 0.0010

Model performance metrics and visualizations can be found in the corresponding notebook and log files.

## Contributors
- Nate Hu, Victor Ji, Tom Suo, Max Jiang

## References

```
@article{senger2024karrierewege,
  title={KARRIEREWEGE: A Large Scale Career Path Prediction Dataset},
  author={Senger, Elena and Campbell, Yuri and Van Der Goot, Rob and Plank, Barbara},
  journal={arXiv preprint arXiv:2412.14612},
  year={2024}
}
```
