const filter = {
    url: [
        {
            urlMatches: 'https://opensea.io/assets/*/*/*',
        },
    ],
};

chrome.webNavigation.onHistoryStateUpdated.addListener(({tabId}) => {
    chrome.scripting.insertCSS(
        {
            target: {tabId},
            files: ["styles.css"],
        },
    );

    chrome.scripting.executeScript(
        {
            target: {tabId},
            files: ['content-script.js'],
        },
    );
}, filter);
