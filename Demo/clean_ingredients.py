
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:02:59 2021

@author: safiu
"""
import pandas as pd




def readIngredientsDictionary(file):

    ing2idx = dict()
    count_ing = 0
    with open(file, 'r') as file:
        for line in file:
            line = line.rstrip('\n').split(',')
            for ing in line:
                ing = ing.strip().lower()
                if ing not in ing2idx.keys():
                    ing2idx[ing] = count_ing
                    count_ing += 1

    idx2ing = {v:k for k,v in ing2idx.items()}

    #return ing2idx, idx2ing
    return ing2idx.keys()



def readBlacklist(file):

    blacklist = []
    with open(file, 'r') as file:
        for line in file:
            line = line.rstrip('\n').strip().lower()
            blacklist.append(line)

    return blacklist



def readBaseIngredients(file):

    base_ingredients = []
    with open(file, 'r') as file:
        for line in file:
            line = line.rstrip('\n').split(',')
            for ing in line:
                ing = ing.strip().lower()
                base_ingredients.append(ing)

    return base_ingredients

def buildIngredientsMapping(ingredients, blacklist, base_ingredients=None):

    ing_mapping = dict()
    new_ing = []
    # Iterate over each ingredient
    for ing in ingredients:
        old_ing = ing.strip()

        # Clean ingredient name with all blacklist terms
        ing_parts = ing.split()
        for b in blacklist:
            if b in ing_parts:
                pos_b = ing_parts.index(b)
                ing_parts = ing_parts[:pos_b]+ing_parts[pos_b+1:]
        ing = ' '.join(ing_parts).strip()

        # Simplify ingredients if contained in base_ingredients list
        found = False
        i = 0
        while not found and i < len(base_ingredients):
            if base_ingredients[i] in ing:
                ing = base_ingredients[i]
                found = True
            i += 1

        # Found a new basic ingredient
        if ing not in new_ing:
            new_ing.append(ing)
            idx = len(new_ing)-1
        else: # Found a matching with an already existent basic ingredient
            idx = new_ing.index(ing)

        # Insert in mapping
        ing_mapping[old_ing] = idx

    return new_ing, ing_mapping


def generateSimplifiedAnnotations(classes_path, in_list, out_list, clean_list, mapping):

    simplified_ingredients = []
    recipes_in = []
    classes = [] 
    with open(in_list, 'r') as in_list:
        for line in in_list:
            line = line.rstrip('\n').split(',')
            # Store all simplified ingredients for each recipe in the list
            simplified_ingredients.append([clean_list[mapping[ing.lower().strip()]] for ing in line])

    with open(classes_path, 'r') as classes_c:
        for clss in classes_c:
            clss = clss.rstrip('\n')
            # Store all simplified ingredients for each recipe in the list
            classes.append(clss)


    with open(out_list, 'w') as out_list:
        for recipe in simplified_ingredients:
            recipe = [ing for ing in recipe if ing]
            recipes_in.append(';'.join(recipe))
            recipe = ','.join(recipe)+'\n'
            pd.DataFrame()
            out_list.write(recipe)
            
    d = {'f_class':classes,'core_ingr':recipes_in}
    df = pd.DataFrame(d)
    
    return df , simplified_ingredients
    
def remove_emty(test_list):
    while("" in test_list) :
        test_list.remove("")
    return test_list 
    

ingredients = readIngredientsDictionary('../annotations/ingredients.txt')
print('Unique ingredients:',len(ingredients))
blacklist = readBlacklist('blacklist.txt')
print('Blacklist terms:',len(blacklist))
base_ingredients = readBaseIngredients('baseIngredients.txt')
print('Base ingredients:',len(base_ingredients))
clean_ingredients_list, raw2clean_mapping = buildIngredientsMapping(ingredients, blacklist,
                                                                    base_ingredients=base_ingredients)
print('Clean ingredients:',len(clean_ingredients_list))

#print clean_ingredients_list

# Generate training files for simplified list of ingredients
classes = '../annotations/classes.txt'
input_all_list = '../annotations/ingredients.txt'
output_all_list = '../annotations/ingredients_simplified.txt'
print('Writing simplified list of ingredients...')
df , simplified_ingredients =generateSimplifiedAnnotations(classes, input_all_list, output_all_list, clean_ingredients_list, raw2clean_mapping)


def get_relative_score(check_recipe  , chosen_recipe   )  : 
    print('ch', check_recipe)
    f_class = check_recipe['f_class']
    check_recipe = check_recipe['core_ingr']
    row_details = []
    
    check_recipe = check_recipe.replace(';' , ' ')
    
    print('check recipe : ' , check_recipe)
    print('chosen recipe :  ' ,  chosen_recipe)
    
    
    document_1_words = chosen_recipe.split()
    document_2_words = check_recipe.split()
    
    common = list(set(document_1_words).intersection( set(document_2_words) ))
    common_score = len(common)
    
    common = ','.join(common)
    
    additional_ingredients = list(set(document_2_words).difference( set(document_1_words) ))
    additional_score = len(additional_ingredients)

    additional_ingredients=','.join(additional_ingredients)

    row_details=[f_class , common , additional_ingredients , common_score , additional_score]
    print(row_details)
    return(row_details)

chosen_dish = 'apple_pie' 
chosen_recipe = df[df['f_class']==chosen_dish]['core_ingr'][0]
chosen_recipe=chosen_recipe.strip()
chosen_recipe = chosen_recipe.replace(';' , ' ')

rd = df.apply(get_relative_score, chosen_recipe = chosen_recipe , axis = 1 )
common_ingredients_dataframe = pd.DataFrame(list(rd) , columns = ['f_class','common_ingredients','additional_ingredients','common_score', 'additional_score'])


#Running Ingredients with TF-IDF
 