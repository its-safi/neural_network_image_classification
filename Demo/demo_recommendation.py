# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 17:38:01 2021

@author: Hatricano
"""

import numpy as np

from tensorflow.keras.preprocessing.image import load_img
import pandas as pd

import os

def open_images(paths):
    '''
    Opens a batch of images, given the image path(s) as a list
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(224,224))
        image = np.array(image)/255.0
        images.append(image)
    return np.array(images)



test_dir = os.path.join(os.getcwd() , 'previous_meal_2\\')
test_paths = []

for label in os.listdir(test_dir):
        test_paths.append(os.path.join(test_dir,label))

saved_models_path = os.path.join(os.getcwd(),'saved_models','model_ResNet_fit_101.h5')
from tensorflow.keras.models import load_model
ressnet50_10classes = load_model(saved_models_path)

predicted_vals = []
for x in range(0,len(test_paths)) : 
    images = open_images([test_paths[x]])
    predicted = ressnet50_10classes.predict(images)[0]
    predicted = np.argmax(predicted)
    print('result : ' , x, predicted)
    predicted_dish_index=predicted
    predicted_vals.append([x,predicted])
    
    
pdf = pd.DataFrame(predicted_vals , columns = (['image id','predicted_class']))

##########################################################################################################

ingredients_file = os.path.join(os.getcwd(), 'Input' , 'ingredients_list.csv')
df=pd.read_csv(ingredients_file)

def get_relative_score(check_recipe  , chosen_recipe   )  : 
    f_class = check_recipe['f_class']
    check_recipe = check_recipe['core_ingr']
    row_details = []
    check_recipe = check_recipe.replace(';' , ' ')
    document_1_words = chosen_recipe.split()
    document_2_words = check_recipe.split()
    common = list(set(document_1_words).intersection( set(document_2_words) ))
    common_score = len(common)
    common = ','.join(common)
    
    additional_ingredients = list(set(document_2_words).difference( set(document_1_words) ))
    additional_score = len(additional_ingredients)
    additional_ingredients=','.join(additional_ingredients)
    
    ease_of_making = common_score/(common_score+additional_score)
    
    row_details=[f_class , common , additional_ingredients , common_score , additional_score , ease_of_making]
    return(row_details)

food_classes = list(df['f_class'])



chosen_dish = food_classes[predicted_dish_index] 
chosen_recipe = df[df['f_class']==chosen_dish]['core_ingr'][predicted_dish_index]
chosen_recipe=chosen_recipe.strip()
chosen_recipe = chosen_recipe.replace(';' , ' ')

rd = df.apply(get_relative_score, chosen_recipe = chosen_recipe , axis = 1 )
common_ingredients_dataframe = pd.DataFrame(list(rd) , columns = ['f_class','common_ingredients','additional_ingredients','common_score', 'additional_score' , 'ease_of_making'])

ingredient_recommendation = common_ingredients_dataframe.sort_values(['ease_of_making'] , ascending = False)

chosen_dish_ingredients = ingredient_recommendation[ingredient_recommendation['f_class'] == chosen_dish]
ingredient_recommendation = ingredient_recommendation[ingredient_recommendation['f_class'] != chosen_dish]

chosen_dish_ingredients = list(chosen_dish_ingredients['common_ingredients'].values)


number_of_top_dishes = 5

ingredient_recommendation = ingredient_recommendation.head(number_of_top_dishes)

ingredient_recc_op1 = ingredient_recommendation[['f_class','common_score','additional_score']]
ingredient_recc_op1
############################################################################################################

import pandas as pd
#####NUTRITION BASED CALCULATION
gender = 'female'
weight = 80
height = 183
age = 26 
activity_coef = 1.55

if(gender == 'Male'): 
    bmr = 655.1 + (9.653 * weight) + (1.850 * 183) - (4.676 * age)
else:
    bmr = 66.47 + (13.75 * weight) + (5.003 * 183) - (6.755 * age)
    
calories_required = bmr * activity_coef

protein_required = (65/2000) * calories_required
carb_required = (280/2000 ) * calories_required
fats_required = (60/2000) * calories_required

nutr_df = pd.read_csv('nutrition_df2.csv', index_col = 'Unnamed: 0')
nutr_df = nutr_df.fillna('0g')
nutr_df.index = nutr_df.index.set_names(['nutrient'])
nutr_df= nutr_df.reset_index()
nutr_df = nutr_df[nutr_df['nutrient'].isin(['Calories','Protein','Carbohydrates','Fat'])]


predicted_dish_index_1 = 34
predicted_dish_index_2 = 45

predicted_dish_array = [34, 45]

nutr_df.columns = ['nutrient'] + food_classes
nutr_df = nutr_df.T
new_header = nutr_df.iloc[0]
nutr_df = nutr_df[1:]
nutr_df.columns = new_header
nutr_df.index = nutr_df.index.set_names(['f_class'])
nutr_df=nutr_df.reset_index()

cal_d1 = 0
fat_d1 = 0
prot_d1 = 0
carb_d1 = 0

import re
def remove_characters(rawdata):
    rawdata2 = re.sub('[^0-9,.]', '', rawdata)
    return float(rawdata2)

dishes_chosen = []
for dish_index in predicted_dish_array:
    chosen_dish = food_classes[dish_index] 
    dishes_chosen.append(chosen_dish)
    cal_d1 = cal_d1 + remove_characters(nutr_df[nutr_df['f_class'] == chosen_dish].Calories.values[0])
    fat_d1 = fat_d1 + remove_characters(nutr_df[nutr_df['f_class'] == chosen_dish].Fat.values[0])
    prot_d1 = prot_d1 + remove_characters(nutr_df[nutr_df['f_class'] == chosen_dish].Protein.values[0])
    carb_d1 = carb_d1 + remove_characters(nutr_df[nutr_df['f_class'] == chosen_dish].Carbohydrates.values[0])

filt_nutr_df = nutr_df[~nutr_df['f_class'].isin(dishes_chosen)]

result_set = pd.DataFrame(columns = ['f_class','cal_deficit','protein_deficit','carb_deficit','fat_deficit' , 'best_meal'])

i=0
for index, row in filt_nutr_df.iterrows():
    cal_deficit = calories_required - cal_d1 - remove_characters(row.Calories)
    protein_deficit = protein_required - prot_d1 - remove_characters(row.Protein)
    carb_deficit = carb_required - carb_d1 - remove_characters(row.Carbohydrates)
    fat_deficit = fats_required - fat_d1 - remove_characters(row.Fat)
    best_meal = abs(protein_deficit) + abs(carb_deficit) + abs(fat_deficit)
    result_set.loc[i] = [row.f_class , cal_deficit , protein_deficit , carb_deficit ,fat_deficit , best_meal ]
    i+=1

    