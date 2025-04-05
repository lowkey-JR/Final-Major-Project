fetch("recommendations.txt")
  .then(response => {
    if (!response.ok) throw new Error("Failed to load");
    return response.text();
  })
  .then(text => {
    document.getElementById("recommendationsBox").textContent = text;
  })
  .catch(error => {
    document.getElementById("recommendationsBox").textContent = "Could not load recommendations.";
    console.error("Fetch error:", error);
  });


document.addEventListener("DOMContentLoaded", () => {
	// BiLSTM Graph
	document.querySelector("#bilstmChart").innerHTML = `
		<img src="static/Bilstm.png" alt="BiLSTM" style="width:100%; max-height:300px;">
	`;

	// AdaBoost Graph
	document.querySelector("#adaboostChart").innerHTML = `
		<img src="static/adaboost.png" alt="AdaBoost" style="width:100%; max-height:300px;">
	`;

	// Random Forest Graph
	document.querySelector("#randomForestChart").innerHTML = `
		<img src="static/random_forest.png" alt="Random" style="width:100%; max-height:300px;">
	`;

	// Logistic Regression Graph
	document.querySelector("#logisticRegressionChart").innerHTML = `
		<img src="static/logestic.png" alt="Logistic Regression" style="width:100%; max-height:300px;">
	`;

	// XGBoost Graph
	document.querySelector("#xgboostChart").innerHTML = `
		<img src="static/xgboost.png" alt="XGBoost" style="width:100%; max-height:300px;">
	`;

	// CatBoost Graph
	document.querySelector("#catboostChart").innerHTML = `
		<img src="static/catboost.jpg" alt="CatBoost" style="width:100%; max-height:300px;">
	`;

	// SVM Graph
	document.querySelector("#svmChart").innerHTML = `
		<img src="static/svm.jpg" alt="SVM" style="width:100%; max-height:300px;">
	`;
});
