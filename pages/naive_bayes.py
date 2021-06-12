import pandas as pd
from modules.naivebayes import NaiveBayes # In streamlit, all imports are relative to root

def run(st):
    st.title("Naive Bayes Classifier")
    st.subheader("By [@JuliusAlphonso](https://twitter.com/JuliusAlphonso) | Fork on [GitHub](https://github.com/JadeMaveric/MachineLearningViz)")

    dataset = st.sidebar.file_uploader("Choose a file")
    probability_selector = st.sidebar.selectbox("Probability Estimate", ['simple', 'm-estimate', 'laplace'])

    if dataset is not None:
        df = pd.read_csv(dataset)
    else:
        st.header("Usage Instructions")
        st.markdown("""
                1. From the sidebar, select a csv file to get started
                2. Also select a probability estimate
                3. Wait for the classifier to finish training fitting onto the data
                4. Once the training is complete, you can explore the calculated probabilities
                5. Use the form generated to predict classes for unseen examples
                """
        )
        st.header("Don't have a dataset? Load a demo")
        demosets = {
            'Tennis': 'https://raw.githubusercontent.com/JadeMaveric/NaiveBayesViz/main/data/tennis.csv',
            'Cars': 'https://raw.githubusercontent.com/JadeMaveric/NaiveBayesViz/main/data/cars.csv',
            'Customers': 'https://raw.githubusercontent.com/JadeMaveric/NaiveBayesViz/main/data/customers.csv'
        }

        dataset = st.selectbox('Dataset', ['None']+list(demosets.keys()))
        
        if dataset != 'None':
            df = pd.read_csv(demosets[dataset])
        else:
            df = None


    if df is not None:
        nb = NaiveBayes()
        nb.fit(df, probability_selector)

        with st.beta_expander("Dataset", expanded=True):
            st.header("Dataset")
            st.write(df)

        with st.beta_expander("Probabilities", expanded=False):
            st.subheader("Prior -- P(Y==outcome)")
            st.write(nb.prior)
            st.subheader("Evidence -- P(X==value)")
            st.write(nb.evidence)
            st.subheader("Likelihood -- P(X==value|Y==outcome)")
            st.write(nb.likelihood)

        st.header("Classify a record")
        with st.form(key='test_record'):
            attribs = [
                st.selectbox(attrib, list(df[attrib].unique()))
                for attrib in nb.attribs
            ]
            submit_button = st.form_submit_button(label='Submit')
        
        if submit_button:
            ans = nb.predict(attribs)
            sorted_ans = sorted(ans, key=ans.__getitem__, reverse=True)
            ans = {key: ans[key] for key in sorted_ans}

            st.write(pd.DataFrame(ans, index=[0]))

