# import the libraries required
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import scipy
import scipy as sc
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl

from mlxtend.frequent_patterns import association_rules
import itertools

import arff
import pandas as pd

# from dotenv import load_dotenv
from flask import (
    Flask,
    render_template,
    redirect,
    request
)
from mlxtend.frequent_patterns import (
    apriori,
    association_rules,
)
from scipy.io import arff
import itertools
from flask import jsonify 

app = Flask(__name__)


df = pd.read_csv("C:/Users/karth/OneDrive/Desktop/clickStreams.csv")

df.category.value_counts(normalize=True)[:10]
df = df.groupby(["uuid","category"]).size().reset_index(name="Count")
basket = (df.groupby(['uuid', 'category'])['Count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('uuid'))
basket.head()

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

basket_sets = basket.applymap(encode_units)
frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
itemset_count=len(frequent_itemsets)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules_count = len(rules)
rules.sort_values("confidence", ascending = False, inplace = True)
rules.head(10)

items = sorted(
    set(itertools.chain.from_iterable(frequent_itemsets.itemsets.values)))
basket = set()



@app.route('/recom', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # print(request.form.getlist('category'), request.form.getlist('items'))
        form_items = request.form.getlist('items')
        basket.update(form_items)

    recommendations = set()
    if basket:
        frozen_basket = frozenset(basket)
        rules_filtered = rules[rules['antecedents'] == frozen_basket]
        consequents = rules_filtered.tail(8)['consequents'].values

        recommendations.update(
            set(itertools.chain.from_iterable(consequents))
        )

    context = {
        'itemset_count': itemset_count,
        'rules_count': rules_count,
        'items': items,
        'basket': basket,
        'recommendations': recommendations,
    }
    return render_template('indexs.html',**context)


@app.route('/reset-basket/', methods=['POST'])
def reset_basket():
    global basket
    basket = set()
    return redirect('/recom')
                                                    
if __name__ == '__main__':
	app.run(debug=True)  

