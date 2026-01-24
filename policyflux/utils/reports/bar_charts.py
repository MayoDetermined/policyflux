import matplotlib.pyplot as plt

def craft_a_bar(data: list[float, int], labels: list[str], title: str, xlabel: str, ylabel: str) -> None:
    """
    Generate and display a bar chart.
    
    Args:
        data: List of numerical values for bars
        labels: Corresponding labels for each bar
        title: Title of the bar chart
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
    """
    plt.figure(figsize=(10, 6))
    plt.bar(labels, data, color='skyblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()