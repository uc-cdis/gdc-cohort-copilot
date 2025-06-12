document.getElementById("submitBtn").addEventListener("click", () => {
  const inputText = document.getElementById("inputText").value;
  const button = document.getElementById('submitBtn');
  button.disabled = true;
  button.classList.add('loading');

  fetch("/api/cohort", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text: inputText })
  })
    .then(response => {
      if (!response.ok) {
        throw new Error("Network response was not ok.");
      }
      return response.json();
    })
    .then(data => {
      document.getElementById("outputText").value = data.cohort || "No cohort in response.";
    })
    .catch(error => {
      document.getElementById("outputText").value = "Error: " + error.message;
    })
    .finally(() => {
      button.disabled = false;
      button.classList.remove('loading');
    });
});
