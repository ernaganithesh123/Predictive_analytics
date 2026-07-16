import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import streamlit as st
import pickle

# Dataset...
data = pd.read_csv("customer_churn_cleaned.csv")

#Sidebar of this project..
page = st.sidebar.selectbox("Select Page",['Home','EDA operation','ML Prediction'])

# Home page of this project...
if page == "Home":
    st.title("Predictive Analytics for Customer Behavior")
    st.divider()
    st.header("Problem Statment")
    st.write("This project focuses on analyzing customer data to understand behavior patterns and predict future actions such as purchases, churn, or engagement. Students will apply data analytics and machine learning techniques to extract insights and build predictive models that support business decision-making. The project is divided into two phases to ensure a systematic approach to learning and implementation.")
    st.divider()
    st.subheader("Basic information about Dataset.")
    st.subheader("Name of the Dataset: customer_churn_cleaned.csv")
    st.dataframe(data.head(10))
    st.divider()
    st.write("To analyze the each column select the required column from the selectbox.")
    select_box = st.selectbox(label="Select a Column:",options=data.columns)
    if select_box == 'CustomerID':
        st.markdown("It is a customerID which defines the customer Identity.")
    elif select_box == 'Age':
        st.markdown("The Age of the customer is Defined.")
    elif select_box == 'Gender':
        st.markdown("The self-identified gender of the customer.")
    elif select_box == 'Tenure':
        st.markdown("How long the customer has been with the company (usually measured in months or years).")
    elif select_box == 'Usage Frequency':
        st.markdown("How often the customer interacts with or uses the service within a specific timeframe.")
    elif select_box == 'Support Calls':
        st.markdown("The total number of times the customer has contacted customer service for assistance.")
    elif select_box == 'Payment Delay':
        st.markdown("The number of days a customer's payment is overdue past the official billing deadline.")
    elif select_box == 'Subscription Type':
        st.markdown("The specific tier or plan level the customer purchased (e.g., Basic, Standard, Premium).")
    elif select_box == 'Contract Length':
        st.markdown("The duration of the customer's legal commitment to the service (e.g., Monthly, Annual).")
    elif select_box == 'Total Spend':
        st.markdown("The cumulative amount of money the customer has paid to the company over time.")
    elif select_box == 'Last Interaction':
        st.markdown("The number of days since the customer last logged in or communicated with the business.")
    else:
        st.markdown("A binary indicator (Yes/No or 1/0) showing whether the customer canceled their service.")   
    st.divider()
    with st.expander("Click to view the Dataset description"):
        st.caption("The Dataset contains 4,40,832 Observations with one null value for each observation.")
        st.write("Data shape:")
        st.caption(data.shape)
        st.write("Data Info:")
        st.dataframe(data.dtypes)
    st.markdown("To view more details, click the **EDA operation** on sidebar.")

# EDA operation of this project...
elif page == 'EDA operation':
    st.title("Customer Churn Prediction: Exploratory Data Analysis")
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("👥 Total Customers", data.shape[0])
    col2.metric("📊 Total Features", data.shape[1])
    col3.metric("🔄 Churned Customers", len(data[data['Churn'] == 1]))
    col4.metric("✅ Non-Churned Customers", len(data[data['Churn'] == 0]))

    st.divider()
    # univarient analysis of the dataset
    st.subheader("Univarient Analysis")
    st.divider()
    # -------------- Age --------------
    st.subheader("Age Distribution of customers")
    st.bar_chart(data.Age.value_counts().head(10))
    st.dataframe(data['Age'].describe())
    st.markdown("""
The age distribution indicates that customers are spread across various age groups, with **40–50 years** being the most common. The average customer age is approximately **39 years**, with ages ranging from **18 to 65 years**. The median age is **39 years**, suggesting that the distribution is fairly centered around middle-aged customers.
""")
    st.divider()
    # -------------- Gender -------------
    st.subheader("Gender Distribution of customers")
    col1, col2 = st.columns(2)
    col1.metric("👤 Male Customers", len(data[data['Gender'] == "Male"]))
    col2.metric("👩 Female Customers", len(data[data['Gender'] == "Female"]))
    sns.countplot(x=data['Gender'])
    st.pyplot(plt)
    st.markdown("""Male customers account for 56.8% of the dataset, while female customers account for 43.2%, making males the majority customer group.""")
    st.divider()
    # -------------- Tenure -------------
    st.subheader("Tenure Distribution of customers")
    col1 , col2 , col3 = st.columns(3)
    col1.metric("📅 Average Tenure", int(data['Tenure'].mean()))
    col2.metric("📉 Minimum Tenure", data['Tenure'].min())
    col3.metric("📈 Maximum Tenure", data['Tenure'].max())
    st.bar_chart(data.Tenure.value_counts().head(10))
    st.dataframe(data['Tenure'].describe())
    st.markdown("""The average customer tenure is approximately 31 months, with tenure ranging from 1 to 60 months. Most customers have been associated with the company for around 32 months, indicating a balanced distribution of customer experience levels.""")
    st.divider()
    # -------------- Usage Frequency -------------
    st.subheader("Usage Frequency Distribution of customers")
    col1 , col2 , col3 = st.columns(3)
    col1.metric("📊 Average Usage Frequency", int(data['Usage Frequency'].mean()))
    col2.metric("📉 Minimum Usage Frequency", data['Usage Frequency'].min())
    col3.metric("📈 Maximum Usage Frequency", data['Usage Frequency'].max())
    st.bar_chart(data['Usage Frequency'].value_counts().head(10))
    st.dataframe(data['Usage Frequency'].describe())
    st.markdown("""The average usage frequency is approximately 15 interactions per month, with a range from 1 to 30. Most customers interact with the service around 5 times per month, indicating a moderate level of engagement.""")
    st.divider()
    # -------------- Support Calls -------------
    st.subheader("Support Calls Distribution of customers")
    col1 , col2 , col3 = st.columns(3)
    col1.metric("📞 Average Support Calls", int(data['Support Calls'].mean()))
    col2.metric("📞 Minimum Support Calls", int(data['Support Calls'].min()))
    col3.metric("📞 Maximum Support Calls", int(data['Support Calls'].max()))
    st.bar_chart(data['Support Calls'].value_counts().head(10))
    st.dataframe(data['Support Calls'].describe())
    st.markdown("""The average number of support calls is approximately "3" per month, with a range from "0 to 10". Most customers make around 1 support call per month, indicating that while some customers require assistance, the majority have minimal issues.""")
    st.divider()
    # -------------- Payment Delay -------------
    st.subheader("Payment Delay Distribution of customers")
    col1 , col2 , col3 = st.columns(3)
    col1.metric("💳 Average Payment Delay", int(data['Payment Delay'].mean()))
    col2.metric("💳 Minimum Payment Delay", int(data['Payment Delay'].min()))
    col3.metric("💳 Maximum Payment Delay", int(data['Payment Delay'].max()))
    st.bar_chart(data['Payment Delay'].value_counts().head(10))
    st.dataframe(data['Payment Delay'].describe())
    st.markdown("""Payment delay ranges from 0 to 30 days, with an average delay of approximately 12 days. Half of the customers have payment delays of 12 days or less, indicating that most customers make payments within a relatively short period.""")
    st.divider()
    # ------------- Subscription Type -------------
    st.subheader("Subscription Type Distribution of customers")
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Basic Type Subscriptions", data[data['Subscription Type'] == 'Basic'].shape[0])
    col2.metric("📦 Premium Type Subscriptions", data[data['Subscription Type'] == 'Premium'].shape[0])
    col3.metric("📦 Standard Type Subscriptions", data[data['Subscription Type'] == 'Standard'].shape[0])
    fig, ax = plt.subplots()
    data["Subscription Type"].value_counts().plot(kind="pie",autopct="%1.1f%%",ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)
    st.markdown("""The dataset contains 3 subscription types, with Standard being the most common subscription plan, used by 1,49,128(33.8%) customers.""")
    st.divider()
    # ------------- Contract Length -------------
    st.subheader("Contract Length Distribution of customers")
    col1, col2, col3 = st.columns(3)
    col1.metric("📄 Monthly Contract", data[data['Contract Length'] == 'Monthly'].shape[0])
    col2.metric("📄 Quarterly Contract", data[data['Contract Length'] == 'Quarterly'].shape[0])
    col3.metric("📄 Annual Contract", data[data['Contract Length'] == 'Annual'].shape[0])
    fig , ax = plt.subplots()
    data["Contract Length"].value_counts().plot(kind="pie",autopct="%1.1f%%",ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)
    st.markdown("""The dataset contains 3 contract length categories, with Annual contracts being the most common, chosen by 1,77,198(40.2%) customers.""")
    st.divider()
    # ------------- Total Spend -------------
    st.subheader("Total Spend Distribution of customers")
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Average Total Spend", int(data['Total Spend'].mean()))
    col2.metric("💰 Minimum Total Spend", int(data['Total Spend'].min()))
    col3.metric("💰 Maximum Total Spend", int(data['Total Spend'].max()))
    st.dataframe(data['Total Spend'].describe())
    st.markdown("""Customer spending ranges from 100 to 1000 units, with an average total spend of approximately 632 units. The median spending is 661 units, indicating that customers generally have a moderate spending pattern.""")
    st.divider()
    # ------------- Last Interaction -------------
    st.subheader("Last Interaction Distribution of customers")
    col1 , col2 , col3 = st.columns(3)
    col1.metric("📊 Average interaction days",int(data['Last Interaction'].mean()))
    col2.metric("⬇️ Minimum interaction days",int(data['Last Interaction'].min()))
    col3.metric("⬆️ Maximum interaction days",int(data['Last Interaction'].max()))
    st.dataframe(data['Last Interaction'].describe())
    st.markdown("""The last interaction period ranges from 1 to 30 days, with an average of approximately 14.5 days. The median value of 14 days indicates that most customers have interacted with the company within the past two weeks.""")
    st.divider()
    # ------------- Churn -------------
    st.subheader("Target column Analysis(Churn)")
    col1,col2 = st.columns(2)
    col1.metric("🔄 Churned Customers",data[data['Churn'] == 1].shape[0])
    col2.metric("✅ Non-Churned Customers",data[data['Churn'] == 0].shape[0])
    # In detail analysis based on gender.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👨 Churned Male",data[(data['Churn'] == 1) & (data['Gender'] == 'Male')].shape[0])
    col2.metric("👩 Churned Female",data[(data['Churn'] == 1) & (data['Gender'] == 'Female')].shape[0])
    col3.metric("👨 Non-Churned Male",data[(data['Churn'] == 0) & (data['Gender'] == 'Male')].shape[0])
    col4.metric("👩 Non-Churned Female",data[(data['Churn'] == 0) & (data['Gender'] == 'Female')].shape[0])
    st.divider()
    
    # Bivarient analysis of the dataset
    
    st.subheader("Bivarient Analysis")
    st.divider()
    # ------------- Age vs Churn -------------
    st.subheader("Age vs Churn")
    col1,col2 = st.columns(2)
    col1.metric("🧓 Average Age of Churned Customers",int(data[data['Churn'] == 1]['Age'].mean()))
    col2.metric("🧑 Average Age of Non-Churned Customers", int(data[data['Churn'] == 0]['Age'].mean()))
    st.dataframe(data.groupby('Churn')['Age'].agg(['mean','median','std','min','max']))
    fig , ax = plt.subplots()
    sns.boxplot(x='Churn',y='Age',data=data,ax=ax)
    st.pyplot(fig)
    st.markdown("""Customer age appears to be associated with churn behavior. Churned customers have a higher average age (41.7 years) compared to retained customers (36.3 years). The median age also follows a similar pattern (42 vs 37 years), suggesting that older customers are more likely to churn than younger customers.""")
    st.divider()
    # ------------- Tenure vs Churn -------------
    st.subheader("Tenure vs Churn")
    col1,col2 = st.columns(2)
    col1.metric("⏳ Average Tenure of Churned Customers",int(data[data['Churn'] == 1]['Tenure'].mean()))
    col2.metric("⏳ Average Tenure of Non-Churned Customers", int(data[data['Churn'] == 0]['Tenure'].mean()))
    st.dataframe(data.groupby('Churn')['Tenure'].agg(['mean','median','std','min','max']))
    fig , ax = plt.subplots()
    sns.boxplot(x='Churn',y='Tenure',data=data,ax=ax)
    st.pyplot(fig)
    st.markdown("""Customer tenure shows a significant relationship with churn. Retained customers have a higher average tenure (32.3 months) compared to churned customers (30.5 months).""")
    st.divider()
    # ------------- Usage Frequency vs Churn -------------
    st.subheader("Usage Frequency vs Churn")
    col1 , col2 = st.columns(2)
    col1.metric("📊 Average Usage Frequency of Churned Customers", int(data[data['Churn'] == 1]['Usage Frequency'].mean()))
    col2.metric("📊 Average Usage Frequency of Non-Churned Customers", int(data[data['Churn'] == 0]['Usage Frequency'].mean()))
    st.dataframe(data.groupby('Churn')['Usage Frequency'].agg(['mean','median','std','min','max']))
    fig, ax = plt.subplots()
    sns.boxplot(x='Churn', y = 'Usage Frequency', data=data, ax=ax)
    st.pyplot(fig)
    st.markdown("""Usage frequency is significantly associated with customer churn. Retained customers have a higher average usage frequency (16.26) compared to churned customers (15.46).""")
    st.divider()
    # ------------- Support Calls vs Churn -------------
    st.subheader("Support Calls vs Churn")
    col1, col2 = st.columns(2)
    col1.metric("📞 Average Support calls of Churned Customers", int(data[data['Churn'] == 1]['Support Calls'].mean()))
    col2.metric("📞 Average Support calls of Non-Churned Customers", int(data[data['Churn'] == 0]['Support Calls'].mean()))
    st.dataframe(data.groupby('Churn')['Support Calls'].agg(['mean', 'median', 'std', 'min', 'max']))
    fig , ax = plt.subplots()
    sns.boxplot(x = 'Churn' , y = 'Support Calls' , data = data, ax = ax)
    st.pyplot(fig)
    st.markdown("""Support Calls show a strong relationship with customer churn. Churned customers made an average of 5.14 support calls, whereas retained customers made only 1.59 support calls.""")
    st.divider()
    # ------------- Payment Delay vs Churn -------------
    st.subheader("Payment Delay vs Churn")
    col1, col2 = st.columns(2)
    col1.metric("💳 Average Payment Delay of Churned Customers", int(data[data['Churn'] == 1]['Payment Delay'].mean()))
    col2.metric("💳 Average Payment Delay of Non-Churned Customers", int(data[data['Churn'] == 0]['Payment Delay'].mean()))
    st.dataframe(data.groupby('Churn')['Payment Delay'].agg(['mean', 'median', 'std', 'min', 'max']))
    fig, ax = plt.subplots()
    sns.boxplot(x = 'Churn', y = 'Payment Delay', data = data, ax = ax)
    st.pyplot(fig)
    st.markdown("""Payment Delay has a strong relationship with customer churn. Churned customers have an average payment delay of 15.22 days, compared to 10.02 days for retained customers.""")
    st.divider()
    # ------------- Total Spend vs Churn -------------
    st.subheader("Total Spend vs Churn")
    col1, col2 = st.columns(2)
    col1.metric("💰Average Revenue by Churned Customers", int(data[data['Churn'] == 1]['Total Spend'].mean()))
    col2.metric("💰Average Revenue by Non-Churned Customers", int(data[data['Churn'] == 0]['Total Spend'].mean()))
    st.dataframe(data.groupby('Churn')['Total Spend'].agg(['mean', 'median', 'std', 'min', 'max']))
    fig , ax = plt.subplots()
    sns.boxplot(x = 'Churn', y = 'Total Spend', data = data, ax = ax)
    st.pyplot(fig)
    st.markdown("""Total Spend has a strong relationship with customer churn. Retained customers have an average total spend of 749.95, whereas churned customers have an average spend of only 541.29.""")
    st.divider()
    # ------------- Last Interaction vs Churn -------------
    st.subheader("Last Interaction vs Churn")
    col1, col2 = st.columns(2)
    col1.metric("📊 Average Last Interaction of Churned Customers", int(data[data['Churn'] == 1]['Last Interaction'].mean()))
    col2.metric("📊 Average Last Interaction of Non-Churned Customers", int(data[data['Churn'] == 0]['Last Interaction'].mean()))
    st.dataframe(data.groupby('Churn')['Last Interaction'].agg(['mean','median','std','min','max']))
    fig , ax = plt.subplots()
    sns.boxplot(x = 'Churn', y = 'Last Interaction', data = data, ax = ax)
    st.pyplot(fig)    
    st.markdown("""Last Interaction shows a significant relationship with customer churn. Churned customers have a higher average Last Interaction value (15.60) compared to retained customers (13.01).""")
    st.divider()
    # Relationship between categorical features and Churn.
    st.subheader("Relationship between Categorical Features and Churn")
    st.divider()
    # ------------- Subscription Type vs Churn -------------
    st.subheader("Subscription Type vs Churn")
    # Basic Plan
    Churned_basic = data[(data['Churn'] == 1) & (data['Subscription Type'] == 'Basic')].shape[0]
    NonChurned_basic = data[(data['Churn'] == 0) & (data['Subscription Type'] == 'Basic')].shape[0]
    col1 , col2 = st.columns(2)
    col1.metric(" 💸 Churned Customers With Basic Plan", Churned_basic)
    col2.metric(" 💸 Non-churned Customers With Basic Plan", NonChurned_basic)
    # Standard Plan
    Churned_Standard = data[(data['Churn'] == 1) & (data['Subscription Type'] == 'Standard')].shape[0]
    NonChurned_Standard = data[(data['Churn'] == 0) & (data['Subscription Type'] == 'Standard')].shape[0]
    col1 , col2 = st.columns(2)
    col1.metric(" 💸 Churned Customers With Standard Plan", Churned_Standard)
    col2.metric(" 💸 Non-churned Customers With Standard Plan", NonChurned_Standard)
    # Premium Plan
    Churned_premium = data[(data['Churn'] == 1) & (data['Subscription Type'] == 'Premium')].shape[0]
    NonChurned_premium = data[(data['Churn'] == 0) & (data['Subscription Type'] == 'Premium')].shape[0]
    col1 , col2 = st.columns(2)
    col1.metric(" 💸 Churned Customers with Premium Plan", Churned_premium)
    col2.metric(" 💸 Non-churned Customers with Premium Plan", NonChurned_premium)
    st.divider()
    # ------------- Contract Length vs Churn -------------
    st.subheader("Contract Length vs Churn")
    # Monthly Contract
    Churned_monthly = data[(data['Churn'] == 1) & (data['Contract Length'] == 'Monthly')].shape[0]
    NonChurned_monthly = data[(data['Churn'] == 0) & (data['Contract Length'] == 'Monthly')].shape[0]
    col1 , col2 = st.columns(2)
    col1.metric("📄 Churned Customer with Monthly plan", Churned_monthly)
    col2.metric("📄 Non-churned Customer with Monthly plan", NonChurned_monthly)
    # Quarterly Contract
    Churned_quarterly = data[(data['Churn'] == 1) & (data['Contract Length'] == 'Quarterly')].shape[0]
    NonChurned_quarterly = data[(data['Churn'] == 0) & (data['Contract Length'] == 'Quarterly')].shape[0]
    col1, col2 = st.columns(2)
    col1.metric("📄 Customer Churned with Quarterly plan",Churned_quarterly)
    col2.metric("📄 Non-churned Customer with Quarterly plan",NonChurned_quarterly)
    # Annual contract
    Churned_annual = data[(data['Churn'] == 1) & (data['Contract Length'] == 'Annual')].shape[0]
    NonChurned_annual = data[(data['Churn'] == 0) & (data['Contract Length'] == 'Annual')].shape[0]
    col1, col2 = st.columns(2)
    col1.metric("📄 Customer Churned with Annual plan",Churned_annual)
    col2.metric("📄 Non-churned Customer with Annual plan",NonChurned_annual)
    st.divider()
    st.markdown("""<div style='text-align:center; color:gray; padding:20px;'> <h4>End of Report</h4> </div>""",unsafe_allow_html=True,)
else:
    st.title("Machine Learning Prediction")
    # Load Trained model
    model = pickle.load(open("Preditive_system.sav","rb"))
    # User input
    Age = st.number_input("Age")
    gender = st.selectbox("Gender",["Male","Female"])
    tenure = st.number_input("Tenure")
    usage_frequency = st.number_input("Usage Frequency")
    support_calls = st.number_input("Support Calls")
    payment_delay = st.number_input("Payment_delay")
    subscription_type = st.selectbox("Subscription Type", ["Basic","Premium","Standard"])
    contract_length = st.selectbox("Contract Length",["Annual","Monthly","Quarterly"])
    total_spend = st.number_input("Total Spend")
    last_interaction = st.number_input("Last Interaction")
    # Create Predict button.
    if st.button("Predict"):
        # Creating dataframe from Input data.
        input_df = pd.DataFrame({
            "Age":[Age],
            "Gender":[gender],
            "Tenure":[tenure],
            "Usage Frequency": [usage_frequency],
            "Support Calls": [support_calls],
            "Payment Delay": [payment_delay],
            "Subscription Type":[subscription_type],
            "Contract Length": [contract_length],
            "Total Spend": [total_spend],
            "Last Interaction": [last_interaction]
        })
        input_df["Age_Group"] = pd.cut(input_df["Age"],bins=[18,30,45,65],labels=["Young","Adult","Senior"])
        input_df["Spend_per_tenure"] = (input_df['Total Spend'] / input_df['Tenure'])
        input_df['Support_Calls_Flag'] = (input_df['Support Calls'] > 5 ).astype(int)
        input_df['Payment_Delay_Category'] = pd.cut(input_df['Payment Delay'],bins=[1,10,20,30],labels=['Low','Medium','High'])
        # one-Hotencoding
        input_df = pd.get_dummies(input_df,columns=['Gender','Subscription Type','Contract Length', 'Age_Group', 'Payment_Delay_Category'],drop_first=True)
        # Arranging columns
        training_col = [
        'Age',
        'Tenure',
        'Usage Frequency',
        'Support Calls',
        'Payment Delay',
        'Total Spend',
        'Last Interaction',
        'Spend_per_tenure',
        'Support_Calls_Flag',
        'Gender_Male',
        'Subscription Type_Premium',
        'Subscription Type_Standard',
        'Contract Length_Monthly',
        'Contract Length_Quarterly',
        'Age_Group_Adult',
        'Age_Group_Senior',
        'Payment_Delay_Category_Medium',
        'Payment_Delay_Category_High'
        ]
        input_df = input_df.reindex(columns = training_col,fill_value = 0)
        prediction = model.predict(input_df)
        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to Churn.")
        else:
            st.success("✅ Customer is Likely to Stay")