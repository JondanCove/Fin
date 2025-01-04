// set endpoint and your access key
endpoint = 'timeframe'
access_key = '58aeba9c13584df0b10441392da572c3';

// get the most recent exchange rates via the "live" endpoint:
//https://api.currencylayer.com/timeframe?58aeba9c13584df0b10441392da572c3start_date=2010-01-01&end_date=2020-01-01
$.ajax({
    url: 'https://api.currencylayer.com/' + endpoint + '?access_key=' + access_key + "start_date=2010-01-01&end_date=2020-01-01",   
    dataType: 'jsonp',
    success: function(json) {

        // exchange rata data is stored in json.quotes
        alert(json.quotes.USDGBP);

        // source currency is stored in json.source
        alert(json.source);

        // timestamp can be accessed in json.timestamp
        alert(json.timestamp);

    }
});