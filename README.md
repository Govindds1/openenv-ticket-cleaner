Ticket Cleaner AI Environment
A real-world simulation of a customer support ticketing cleanup task built for the OpenEnv specification.

Overview
This environment simulates the daily task of a data operations specialist cleaning messy support ticket logs. It allows an AI agent to interact with a database to ensure data integrity and standardization.

Action Space
The agent can perform the following commands via the step() API:

remove_duplicates: Deduplicates the dataset based on all columns.

fix_priority: Implements logic to fill missing (null) priority values with a "Medium" default.

standardize_status: Normalizes the 'status' column to lowercase to ensure consistency.

Tasks & Difficulty
Task 1 (Easy): Identify and remove identical duplicate rows.

Task 2 (Medium): Handle missing data by imputing values into the priority column.

Task 3 (Hard): Perform string normalization across the status column.

Setup & Usage
Prerequisites: Docker, Python 3.10+

Build: docker build -t ticket-cleaner .

Run: docker run -p 7860:7860 ticket-cleaner

Evaluate: Run python inference.py to see the agent in action.
