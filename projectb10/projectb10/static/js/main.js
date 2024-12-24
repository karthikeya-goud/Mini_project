$(document).ready(function () {
    $('#fetch-results').click(function () {
        $.ajax({
            url: '/fetch_previous_results/',
            method: 'GET',
            success: function (data) {
                $('#results-list').empty();
                data.forEach(function (result) {
                    $('#results-list').append(`
                        <li>
                            <strong>Video:</strong> <a href="${result.video_url}" target="_blank">Watch Video</a><br>
                            <strong>Detected Actions:</strong> ${result.detected_actions.join(', ')}<br>
                            <strong>Confidence Scores:</strong> ${result.confidence_scores.join(', ')}<br>
                            <strong>Timestamp:</strong> ${result.timestamp}
                        </li>
                    `);
                });
            }
        });
    });
});