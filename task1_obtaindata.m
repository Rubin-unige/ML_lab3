
% Task 1: Obtain a data set

wine_data = load("Data\small_wine\wine.data");

wine_data.Properties.VariableNames = {'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash' , 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols',   'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315', 'Proline'};

disp(wine_data);