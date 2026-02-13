class SimpleTextProcessor:
    """A simple text processor that can be used in the policy flux pipeline."""
    
    def __init__(self, text: str):
        self.text = text
    
    def process(self) -> str:
        """Process the text and return the result."""
        # For demonstration, we'll just return the text in uppercase
        return self.text.upper()