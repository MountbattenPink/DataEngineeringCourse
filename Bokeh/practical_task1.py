## Visualizing Survival Rates Across Different Demographic
# Objective: Create interactive visualizations using Bokeh to explore the survival rates of passengers
# on the Titanic based on various demographic factors such as age, gender, and class.
# Task Requirements:
# 1. Data Preparation:
# o Handle missing values in the Age, Cabin, and Embarked columns appropriately.
# o Create a new column AgeGroup to categorize passengers into age groups (e.g., Child, Young Adult, Adult, Senior).
# o Create a SurvivalRate column to calculate the percentage of passengers who
# survived within each group.
# 2. Visualization:
# o Age Group Survival: Create a bar chart showing survival rates across different age groups.
# o Class and Gender: Create a grouped bar chart to compare survival rates across
# different classes (1st, 2nd, 3rd) and genders (male, female).
# o Fare vs. Survival: Create a scatter plot with Fare on the x-axis and survival status
# on the y-axis, using different colors to represent different classes.
# 3. Interactivity:
# o Add hover tools to display detailed information when hovering over any bar or point.
# o Implement filtering options to allow users to filter visualizations by class or gender.
# 4. Output:
# o Save the visualizations as HTML files that can be viewed in a web browser.
# Execution and Verification:
# • Ensure that the visualizations are interactive and provide meaningful insights.
# • Test the visualizations with different filters to verify their functionality.
# Deliverables:
# • A Python script (.py file) containing the data preparation, Bokeh visualization code, and
# necessary functions.
# • HTML files for each visualization created.