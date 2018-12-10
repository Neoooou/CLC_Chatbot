function syntaxHighlight(json) {
    json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
        var cls = 'number';
        if (/^"/.test(match)) {
            if (/:$/.test(match)) {
                cls = 'key';
            } else {
                cls = 'string';
            }
        } else if (/true|false/.test(match)) {
            cls = 'boolean';
        } else if (/null/.test(match)) {
            cls = 'null';
        }
        return '<span class="' + cls + '">' + match + '</span>';
    });
}
function getAccessToken(){
    url = 'http://localhost:5000/chatbot/login'
    $.ajax({
        url:url,
        type:'POST',
        data:{
            'username':'test',
            'password':'test'
        },
        success:function(json){
            console.log(json)
        }
    })

}

function sendinput() {
    var input_box = $('#chatbox');
    var user_input = input_box.val();
    input_box.val('');
    $.ajax({
        url: "send_user_input",
        type: "GET",
        data: {
            "user_input": user_input
        },
        success: function(json) {
            $('#chatboard').append(
                '<div style="clear:both;float:none"></div><div id="chatbubble" class="user">' + user_input + '</div>'
            );

            $('#chatboard div[id=chatbubble]:last').animate({
                marginRight: '10px'
            }, 100, "swing");

            var chatboard = $('#chatboard');
            var chatHeight = chatboard[0].scrollHeight;
            chatboard.scrollTop(chatHeight);
            setTimeout(function() {

                $('#chatboard').append(
                    '<div id="pending"><div style="clear:both;float:none"></div><div id="chatbubble" class="bot">...</div></div>'
                );

                $('#chatboard div:last').animate({
                    marginLeft: '10px',
                }, 100, "swing");

                var chatboard = $('#chatboard');
                var chatHeight = chatboard[0].scrollHeight;
                chatboard.scrollTop(chatHeight);
            }, 800);
            setTimeout(function() {
                $('#pending').remove();
                var matched_link = json['link']
                $('#chatboard').append(
                    '<div style="clear:both;float:none"></div><div id="chatbubble" class="bot">' + json['res'] + '</div>'
                );
                if(matched_link.length > 0 &&
                    matched_link.indexOf('https://childrenslegalcentre.wales/how-the-law-affects-me/') > -1){
                    $('#clc_frame').attr("src",matched_link);
                    //switchCss();
                    $('#chatboard').append(
                        '<div style="clear:both;float:none"></div><div id="chatbubble" class="bot">' + json['doc'] + '</div>'
                    );
                    window.find("the",false,true);

                }


                $('#chatboard div:last').css('margin-left', '10px');
                var chatboard = $('#chatboard');
                var chatHeight = chatboard[0].scrollHeight;
                chatboard.scrollTop(chatHeight);
            }, 2500);
        }
    });
}
// change the CSS style
function switchCss(){
    $('#button').css({"width": "18%","height": "35px"});
    $('#sendbutton').css({"line-height": "40px"});
    $('#chatbox').css({"width":"80%", "height":"37px"});
    $('#conversation').css({"position": "fixed","right":"14px", "bottom":"4px","width":"27%",
        "height":"45vh","border":"1px solid #e1e8ed", "border-radius":"13px", "padding":"5px 5px 34px 5px"});
    $('#clc_frame').css({"width":"71%", "height":"99vh"});
    $('#chatbubble').css({"width":"240px","padding":"8px"});
    $('#matched_doc').css({"float":"right","width":"28%", "height":"54vh",
        "background-color":"#3A3A3A","padding":"10px 35px 10px 26px",
        "box-sizing":"border-box","color":"white",
        "position":"relative", "overflow-y":"auto"});
}
$(document).keypress(function(e) {
    if((e.which == 13) && (!$('#chatbox').val().length == 0)) {
        $('#sendbutton').addClass("active");
        sendinput();
    }
});

$('#chatbox').keyup(function(e) {
    $('#sendbutton').removeClass("active");
});

$('#sendbutton').on('click', function() {
    if(!$('#chatbox').val().length == 0){
        sendinput();
    }
});

$('#report-button').on('click', function() {
    $.ajax({
        url: "render_pdf",
        type: "GET",
        success: function() {
            // nothing to do
        }
    });
});