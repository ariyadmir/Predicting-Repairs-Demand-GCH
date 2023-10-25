# Predicting-Repairs-Demand-GCH
This project was undertaken and submitted in partial fulfilment of the requirements for Msc Data Science and Analytics at Cardiff University

In summary, the objectives of the project are outlined below: 
• Analyze and examine historical repair and maintenance data, and identify high and low-demand properties
• Determine if repair demand for various types of repairs can be predicted on the basis of weather, time of year, property, and tenant data
• Predict the demand for each repair priority and trade based on time of year and weather data
• Predict the demand for each repair priority and trade based on property characteristics and tenant demographic data
• Create a prototype tool that connects to our predictive models and displays our predicted repair demand for each type of repair, with the ultimate goal of potentially integrating it into GCH’s service framework.

1. The first objective of predicting repair demand based on weather and time of year data was executed in two separate notebooks; one for creating models that predict demand for all types of repair 'priority' and the other for creating models that predict demand for all types of repair 'tasks' (or trades). 

2. The second objective of predicting repair demand based on property data was executed in two separate notebooks as well; one for creating models that predict demand for all types of repair 'priority' and the other for creating models that predict demand for all types of repair 'tasks' (or trades).

3. The models created in these four notebooks were then loaded into a fifth notebook for integrating them into a web-application created in Dash. 


Abstract:
The social housing sector in the UK, along with other public sector industries, is currently undergoing a considerable transition towards the adoption of predictive analysis and modelling for adding value to their service operations. As the previous reliance on resource-heavy responsive and preventative repair service models are no longer optimal in the current economy, steps are being taken by housing associations to embrace data-driven technologies for more effective business strategies. To enhance the repair and maintenance service delivery for our client, Gloucester City Homes, and address any service backlogs, we propose the use of state-of-the-art tree-based boosting algorithms, such as LightGBM and CatBoost, to predict repair demand based on time of year, weather conditions, and property characteristics. We ensure model interpretability by employing contemporary techniques, such as SHAP, which provide localized explanations for predictions within the feature space. We integrate our models into a front-end web application that is designed as a tool for predicting repair demand.

