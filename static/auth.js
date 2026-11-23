// Shared session handling for the logged-in pages (index, harvests, financials, account).
// Load this BEFORE each page's own <script>.

(function () {
    const originalFetch = window.fetch.bind(window);

    // Wrap fetch so an expired or missing session sends the farmer to the login page
    // instead of leaving a half-empty page.
    //  - 401 on any request  -> not logged in / session expired
    //  - 403 on a GET request -> the page asked for another account's data
    //    (localStorage and the session cookie disagree), so start over.
    //  - 403 on POST/PUT is left alone: it's a normal "that isn't yours" error
    //    the page shows in an alert.
    window.fetch = async function (input, init = {}) {
        const res = await originalFetch(input, init);
        const method = (init.method || 'GET').toUpperCase();
        if (res.status === 401 || (res.status === 403 && method === 'GET')) {
            localStorage.removeItem('agritech_user_id');
            window.location.replace('/login');
            // Never resolve, so the calling code stops here instead of
            // trying to render an error response.
            return new Promise(() => {});
        }
        return res;
    };
})();

// Ends the session on the server (deletes the cookie) and returns to login.
async function signOut() {
    try {
        await fetch('/api/auth/logout', { method: 'POST' });
    } catch (e) {
        // Even if the server can't be reached, still clear the browser side.
    }
    localStorage.removeItem('agritech_user_id');
    window.location.replace('/login');
}
