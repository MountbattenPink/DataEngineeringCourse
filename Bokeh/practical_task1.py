import numpy as np
import pandas as pd
from bokeh.plotting import  save
from bokeh.plotting import figure, curdoc, output_file, show
from bokeh.models import  Select
from bokeh.layouts import column

from bokeh.transform import factor_cmap
from bokeh.models import ColumnDataSource, FactorRange, HoverTool

## Visualizing Survival Rates Across Different Demographic
# Objective: Create interactive visualizations using Bokeh to explore the survival rates of passengers
# on the Titanic based on various demographic factors such as age, gender, and class.

csv_path = 'resources/Titanic-Dataset.csv'
data_df = pd.read_csv(csv_path, delimiter=',')
colors = np.array(['navy', 'olive', 'green', 'orange', 'red', 'purple', 'cyan', 'magenta', 'yellow', 'gray'])


# Task Requirements:
# 1. Data Preparation:
# o Handle missing values in the Age, Cabin, and Embarked columns appropriately.
data_df = data_df[data_df['Age'].notna()]
data_df['Cabin'] = 'Unknown' if data_df['Cabin'].isna else data_df['Cabin']
data_df['Embarked'] = 'Unknown' if data_df['Embarked'].isna else data_df['Embarked']


# o Create a new column AgeGroup to categorize passengers into age groups (e.g., Child, Young Adult, Adult, Senior).
def determine_age_group(age):
    if age < 18:
        return 'Child'
    elif age < 25:
        return 'Young Adult'
    elif age < 60:
        return 'Adult'
    else:
        return 'Senior'

data_df['AgeGroup'] = data_df['Age'].apply(determine_age_group)

# o Create a SurvivalRate column to calculate the percentage of passengers who survived within each group.
total_survived = data_df.groupby('AgeGroup')['Survived'].sum()
total_unsurvived = data_df[data_df['Survived'] == 0].groupby('AgeGroup')['Survived'].count()

all_age_groups = pd.DataFrame({
    'Survived': total_survived,
    'Unsurvived': total_unsurvived
}).reset_index()

all_age_groups['Success_rate'] = all_age_groups['Survived'] / (all_age_groups['Survived'] + all_age_groups['Unsurvived'])

# 2. Visualization:
# o Age Group Survival: Create a bar chart showing survival rates across different age groups.

x = all_age_groups['AgeGroup']
y = all_age_groups['Success_rate']
def create_bar_plot(x_df, y_df):
    p = figure(x_range=x_df, height=350, title="Age Group Survival", toolbar_location=None,
               tools='pan,box_zoom,reset,hover', toolbar_sticky=False)
    p.vbar(x=x_df, top=y_df, width=0.9)
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    output_file("resources/2.1.Age Group Survival.html")
    save(p)

create_bar_plot(x, y)



# o Class and Gender: Create a grouped bar chart to compare survival rates across
# different classes (1st, 2nd, 3rd) and genders (male, female).
total_survived_by_class = data_df[['Pclass', 'Sex', 'Survived']].groupby(['Sex','Pclass']).agg(
    survived=('Survived', 'sum'),
    total=('Survived', 'count')).reset_index()
print(total_survived_by_class)

total_survived_by_class['success_rate'] = total_survived_by_class['survived'] / total_survived_by_class['total']
print(total_survived_by_class)


def create_grouped_plot(df):
    df = df.reset_index()
    df['Pclass_Sex'] = df['Pclass'].astype(str) + '-' + df['Sex']
    source = ColumnDataSource(df)
    pclasses = sorted(df['Pclass'].unique())
    genders = sorted(df['Sex'].unique())
    x_labels = [f"{pclass}-{gender}" for pclass in pclasses for gender in genders]
    p = figure(x_range=FactorRange(*x_labels), height=400, width=400,
               title="Success Rate by Class and Gender", toolbar_location=None,
               tools='pan,box_zoom,reset,hover')
    p.vbar(x='Pclass_Sex', top='success_rate', width=0.5, source=source,
           color=factor_cmap('Sex', palette=colors, factors=genders),
           legend_field='Sex')
    hover = HoverTool()
    p.add_tools(hover)
    p.xgrid.grid_line_color = None
    p.xaxis.major_label_orientation = 1
    p.yaxis.axis_label = 'Success Rate'
    p.xaxis.axis_label = 'Class-Gender'
    p.legend.title = 'Sex'
    output_file('resources/2.2.Gender_and_Class_Survival.html')
    show(p)

create_grouped_plot(total_survived_by_class)


# o Fare vs. Survival: Create a scatter plot with Fare on the x-axis and survival status
# on the y-axis, using different colors to represent different classes.
class_to_status = data_df[['Survived', 'Fare', 'Pclass']]
def create_scatter(df):
    p = figure(width=1500, height=800,
               x_axis_label='Fare',
               y_axis_label='Survival Status',
               tools='pan,box_zoom,reset,hover',
               toolbar_sticky=False)
    for i, pclass in enumerate(df['Pclass'].unique()):
        sub_df = df[df['Pclass'] == pclass]
        p.scatter(sub_df['Fare'], sub_df['Survived'], size=5, color=colors[i], alpha=0.5)
    hover = HoverTool()

    p.add_tools(hover)
    p.yaxis.ticker = [0, 1]
    p.yaxis.axis_label = 'Survival Status'
    p.xaxis.axis_label = 'Fare'
    output_file('resources/2.3.Fare_to_Survival.html')
    show(p)

create_scatter(class_to_status)

# 3. Interactivity:
# o Add hover tools to display detailed information when hovering over any bar or point.
#ADDED ABOVE

# o Implement filtering options to allow users to filter visualizations by class or gender.

df = total_survived_by_class.copy()
def create_filtered_bar_plot(df):
    source = ColumnDataSource(df[df['Pclass'] == 2])
    unique_genders = df['Sex'].unique().tolist()
    p = figure(x_range=unique_genders, height=700, width=700, title="Success Rate", tools="")
    p.vbar(x='Sex', top='success_rate', width=0.5, source=source,
       color=factor_cmap('Sex', palette=colors, factors=df['Sex'].unique()),
       legend_field='Sex')
    pclass_select = Select(title="Select Class", value="1", options=[str(i) for i in df['Pclass'].unique()])
    def callback(attr, old, new):
        selected_pclass = int(pclass_select.value)
        filtered_df = df[df['Pclass'] == selected_pclass]
        source.data = ColumnDataSource(filtered_df).data

    pclass_select.on_change('value', callback)

    layout = column(pclass_select, p)
    hover = HoverTool()
    hover.tooltips = [("Gender", "@Sex"), ("Success Rate", "@success_rate")]
    p.add_tools(hover)

    p.xaxis.axis_label = 'Gender'
    p.yaxis.axis_label = 'Success Rate'
    p.xaxis.major_label_orientation = "vertical"
    p.legend.title = 'Gender'

    output_file('resources/3.filtered_by_class.html')
    show(p)

# 4. Output:
# o Save the visualizations as HTML files that can be viewed in a web browser.

# Execution and Verification:
# • Ensure that the visualizations are interactive and provide meaningful insights.
# • Test the visualizations with different filters to verify their functionality.

# Deliverables:
# • A Python script (.py file) containing the data preparation, Bokeh visualization code, and
# necessary functions.
# • HTML files for each visualization created.