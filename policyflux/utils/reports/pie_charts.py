import matplotlib.pyplot as plt

def bake_a_pie(data: list[float, int], labels: list[str], title: str) -> None:
    """
    Generate and display a pie chart.
    
    Args:
        data: List of numerical values for pie slices
        labels: Corresponding labels for each slice
        title: Title of the pie chart
    """
    plt.figure(figsize=(8, 8))
    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()