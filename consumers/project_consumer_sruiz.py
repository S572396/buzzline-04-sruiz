#####################################
# Import Modules
#####################################

# Import packages from Python Standard Library
import os
import json  # handle JSON parsing
from collections import defaultdict  # data structure for counting author occurrences

# Import external packages
from dotenv import load_dotenv

# IMPORTANT
# Import Matplotlib.pyplot for live plotting
# Use the common alias 'plt' for Matplotlib.pyplot
# Know pyplot well
import matplotlib.pyplot as plt

# Import functions from local modules
from utils.utils_consumer import create_kafka_consumer
from utils.utils_logger import logger
from textblob import TextBlob  # Sentiment analysis

#####################################
# Load Environment Variables
#####################################

load_dotenv()

#####################################
# Getter Functions for .env Variables
#####################################

def get_kafka_topic() -> str:
    """Fetch Kafka topic from environment or use default."""
    topic = os.getenv("BUZZ_TOPIC", "unknown_topic")
    logger.info(f"Kafka topic: {topic}")
    return topic

def get_kafka_consumer_group_id() -> str:
    """Fetch Kafka consumer group id from environment or use default."""
    group_id: str = os.getenv("BUZZ_CONSUMER_GROUP_ID", "default_group")
    logger.info(f"Kafka consumer group id: {group_id}")
    return group_id

#####################################
# Set up data structures
#####################################

# Initialize a dictionary to store author counts and sentiments
author_counts = defaultdict(int)
author_sentiments = defaultdict(list)

#####################################
# Set up live visuals
#####################################

# Use the subplots() method to create a tuple containing
# two objects at once:
# - a figure (which can have many axis)
# - an axis (what they call a chart in Matplotlib)
fig, (ax_count, ax_sentiment) = plt.subplots(1, 2, figsize=(12, 6))

# Use the ion() method (stands for "interactive on")
# to turn on interactive mode for live updates
plt.ion()

#####################################
# Define an update chart function for live plotting
# This will get called every time a new message is processed
#####################################

def update_chart():
    """Update the live chart with the latest author counts."""
    # Clear the previous chart
    ax_count.clear()

    # Get the authors and counts from the dictionary
    authors_list = list(author_counts.keys())
    counts_list = list(author_counts.values())

    # Create a bar chart using the bar() method.
    ax_count.bar(authors_list, counts_list, color="skyblue")

    # Set chart labels
    ax_count.set_xlabel("Authors")
    ax_count.set_ylabel("Message Counts")
    ax_count.set_title("Real-Time Author Message Counts")

    # Rotate x-axis labels
    ax_count.set_xticklabels(authors_list, rotation=45, ha="right")

    # Adjust layout and draw the chart
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

def update_sentiment_chart():
    """Update the live sentiment chart with the latest sentiment data."""
    # Clear the previous chart
    ax_sentiment.clear()

    # Get the authors and their average sentiment scores
    authors = list(author_sentiments.keys())
    avg_sentiments = [sum(scores) / len(scores) for scores in author_sentiments.values()]

    # Create a bar chart using the average sentiment scores for each author
    ax_sentiment.bar(authors, avg_sentiments, color="lightgreen")

    # Set chart labels
    ax_sentiment.set_xlabel("Authors")
    ax_sentiment.set_ylabel("Average Sentiment")
    ax_sentiment.set_title("Real-TimeSentiment Distribution By Author")

    # Rotate x-axis labels
    ax_sentiment.set_xticklabels(authors, rotation=45, ha="right")

    # Adjust layout and draw the chart
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)

#####################################
# Sentiment Analysis Function
#####################################

def analyze_sentiment(message: str) -> float:
    """
    Analyze the sentiment of a given message using TextBlob.
    Returns a sentiment polarity score between -1 (negative) and 1 (positive).
    
    Args:
        message (str): The message whose sentiment is to be analyzed.
    """
    blob = TextBlob(message)
    return blob.sentiment.polarity

#####################################
# Function to process a single message
#####################################

def process_message(message: str) -> None:
    """
    Process a single JSON message from Kafka, update author counts, and track sentiment.

    Args:
        message (str): The JSON message as a string.
    """
    try:
        # Log the raw message for debugging
        logger.debug(f"Raw message: {message}")

        # Parse the JSON string into a Python dictionary
        message_dict: dict = json.loads(message)

        # Ensure the processed JSON is logged for debugging
        logger.info(f"Processed JSON message: {message_dict}")

        # Ensure it's a dictionary before accessing fields
        if isinstance(message_dict, dict):
            # Extract the 'author' and 'message' fields from the dictionary
            author = message_dict.get("author", "unknown")
            message_text = message_dict.get("message", "")

            # Analyze sentiment of the message
            sentiment_score = analyze_sentiment(message_text)
            logger.info(f"Sentiment score for message: {sentiment_score}")

            # Track sentiment for the author
            author_sentiments[author].append(sentiment_score)

            # Increment the message count for the author
            author_counts[author] += 1

            # Log the updated counts
            logger.info(f"Updated author counts: {dict(author_counts)}")

            # Update the author counts chart
            update_chart()

            # Update the sentiment distribution chart
            update_sentiment_chart()

            # Log the updated charts
            logger.info(f"Charts updated successfully for message: {message}")
        else:
            logger.error(f"Expected a dictionary but got: {type(message_dict)}")

    except json.JSONDecodeError:
        logger.error(f"Invalid JSON message: {message}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

#####################################
# Define main function for this module
#####################################

def main() -> None:
    """
    Main entry point for the consumer.

    - Reads the Kafka topic name and consumer group ID from environment variables.
    - Creates a Kafka consumer using the `create_kafka_consumer` utility.
    - Polls messages and updates a live chart.
    """
    logger.info("START consumer.")

    # fetch .env content
    topic = get_kafka_topic()
    group_id = get_kafka_consumer_group_id()
    logger.info(f"Consumer: Topic '{topic}' and group '{group_id}'...")

    # Create the Kafka consumer using the helpful utility function.
    consumer = create_kafka_consumer(topic, group_id)

    # Poll and process messages
    logger.info(f"Polling messages from topic '{topic}'...")
    try:
        for message in consumer:
            # message is a complex object with metadata and value
            # Use the value attribute to extract the message as a string
            message_str = message.value
            logger.debug(f"Received message at offset {message.offset}: {message_str}")
            process_message(message_str)
    except KeyboardInterrupt:
        logger.warning("Consumer interrupted by user.")
    except Exception as e:
        logger.error(f"Error while consuming messages: {e}")
    finally:
        consumer.close()
        logger.info(f"Kafka consumer for topic '{topic}' closed.")

    logger.info(f"END consumer for topic '{topic}' and group '{group_id}'.")

#####################################
# Conditional Execution
#####################################

if __name__ == "__main__":

    # Call the main function to start the consumer
    main()

    # Turn off interactive mode after completion
    plt.ioff()  

    # Display the final chart
    plt.show()
