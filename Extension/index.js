const currentUrl = window.location.href;

function checkUrl() {
    // make a POST API call with the current URL as a parameter
    console.log({currentUrl});
    fetch('http://127.0.0.1:8000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ url: currentUrl })
    })
      .then(response => response.json())
      .then(data => {
        console.log(data[0]);
        if (data[0] === 0) {
          alert("Warning: This site may be malicious.");
        }
      })
      .catch(error => console.error(error));
  }
  
// check the URL on initial page load
checkUrl();

// check the URL whenever the browser's URL changes
if ('popstate' in window) {
    window.addEventListener("popstate", checkUrl);
} 
if ('onhashchange' in window) {
    window.addEventListener("hashchange", checkUrl);
}