$(document).ready(function() {
    
    ////// jQueryUI theming //////
    $("#button, #button1, #reset_form, #reset_form1, #hide_search_form, #more_options, .more_context").button();
    $('#button2').button();
    $("#page_num, #field, #method, #year_interval").buttonset();
    $("#word_num").spinner({
        spin: function(event, ui) {
                if ( ui.value > 10 ) {
                    $(this).spinner( "value", 1 );
                    return false;
                } else if ( ui.value < 1 ) {
                    $(this).spinner( "value", 10 );
                    return false;
                }
            }
    });
    $("#word_num").val(5);
    $("#show_search_form").tooltip({ position: { my: "left+10 center", at: "right" } });
    $(".tooltip_link").tooltip({ position: { my: "left top+5", at: "left bottom", collision: "flipfit" } }, { track: true });
    $('#search_explain').accordion({
        collapsible: true,
        heightStyle: "content",
        active: false,
        animate: "swing"
    });
    $('.ui-spinner').css('width', '45px')
    $(':text').addClass("ui-corner-all");
    
    ////////////////////////////////////////
    
    
    // Initialize important variables
    var db_url = db_locals['db_url'];
    var q_string = window.location.search.substr(1);
    
    // Display report tabs according to db.locals.py config
    for (i in db_locals['search_reports']) {
        var search_report = '#' + db_locals['search_reports'][i] + '_button';
        $(search_report).show();
    }
    
    
    $('#more_options').click(function() {
        $('#search_elements').css('z-index', 150);
        $('.book_page').css('z-index', 90);
        if ($(this).text() == "Show search options") {
            showMoreOptions("all");
            $('#search_explain').fadeIn(300); 
        } else {
            hideSearchForm();
        }
    });
    $("#report").buttonset();
    $('#form_body').show();
    if (global_report == "landing_page") {
        showHide('concordance');
        $('#search_explain').fadeIn(300);
        $('#search_elements').css({'max-height': '614px', 'opacity': 100});
        $('#conc_question').fadeIn(300);
    } else {
        $('#initial_form').css({'max-height': '94px', 'opacity': 100});
        $("[id$='_question']").hide();
        $('#search_explain').hide();
        showHide($('input[name=report]:checked', '#search').val());
    }
    $('#report').find('label').click(function() {
        var report = $(this).attr('for');
        if ($("#search_elements").css('max-height') != 0) {
            showHide(report);
            if (report != "frequencies") {
                $("#search_elements").css({'max-height': '614px', 'opacity': 100});
                $('#search_explain').fadeIn(300);
            }
            showMoreOptions();
        } else {
            showHide(report);
            if (report != "frequencies") {
                $("#search_elements").fadeIn();
                $('#search_explain').fadeIn();
            }
            showMoreOptions();
        }
    });
    
    
    // Set-up autocomplete for words and metadata
    autoCompleteWord(db_url);
    for (i in db_locals["metadata_fields"]) {
        var  metadata = $("#" + db_locals["metadata_fields"][i]).val();
        var field = db_locals["metadata_fields"][i];
        autoCompleteMetadata(metadata, field, db_url)
    }
    
    //  This will prefill the search form with the current query
    var val_list = q_string.split('&');
    for (var i = 0; i < val_list.length; i++) {
        var key_value = val_list[i].split('=');
        var value = decodeURIComponent((key_value[1]+'').replace(/\+/g, '%20'));
        var key = key_value[0]
        if (value) {
            if (key == 'report') {
                $('input[name=' + key + '][value=' + value + ']').attr("checked", true);
                $("#report").buttonset("refresh");
            }
            else if (key == 'pagenum' || key == 'method' || key == 'year_interval') {
                $('input[name=' + key + '][value=' + value + ']').attr("checked", true);
                $('input[name=' + key + '][value=' + value + ']').button('refresh');
            }
            else if (key == 'field') {
                $('select[name=' + key + ']').val(value);
            }
            else {
                $('#' + key).val(value);
            }
        }
    }
    
    //  Clear search form
    $("#reset_form, #reset_form1").click(function() {
        $("#method").find("input:radio").attr("checked", false).end();
        $("#method1").attr('checked', true);
        $("#method").buttonset('refresh');
        $("#page_num").find("input:radio").attr("checked", false).end();
        $("#pagenum1").attr('checked', true);
        $("#page_num").buttonset('refresh');
        $('#search')[0].reset();
        $("#search_elements").fadeIn();
        $("#reset_form1").css('color', '#555555 !important');
        $("#report").find("input:radio").attr("checked", false).end();
        $('#concordance').attr('checked', true);
        $('#concordance')[0].click();
    });
    
    //  This is to select the right option when clicking on the input box  
    $("#arg_proxy").focus(function() {
        $("#arg_phrase").val('');
        $("#method1").attr('checked', true).button("refresh");
    });
    $("#arg_phrase").focus(function() {
        $("#arg_proxy").val('');
        $("#method2").attr('checked', true).button("refresh");
    });
    
    $('#syntax').offset({"left":  $('#q').offset().left});
    
    $('#syntax_title').mouseup(function() {
        if ($('#syntax_explain').not(':visible')) {
            $('#syntax_explain').fadeIn('fast');
        }
        $(document).mousedown(function() {
            $('#syntax_explain').fadeOut('fast');
        });
    });
    
//    Check if the search form has any input has been prefilled
    $('input:text').each(function(){
        if ($(this).val().length) {
            if ($(this).attr('id') != 'word_num') {
                $('#reset_form1').find('.ui-button-text').blink()
                return false;
            }
        }
    });
    
    /// Make sure search form doesn't cover footer
    var form_offset = $('#form_body').offset().top + $('#form_body').height();
    searchFormOverlap(form_offset);
    $(window).resize(function() {
        searchFormOverlap(form_offset);
    });
    
    adjustReportWidth();
    adjustBottomBorder();
    
    // Add spinner to indicate that a query is running in the background
    $('#button, #button1').click(function( event ) {
        var width = $(window).width() / 2 - 100;
        hideSearchForm();
        $("#waiting").css("margin-left", width).css('margin-top', 100).show();
    });
});

function searchFormOverlap(form_offset) {
    var footer_offset = $('#footer').offset().top;
    if (form_offset > footer_offset) {
        $('#footer').css('top', form_offset + 20);
    } else {
        $('#footer').css('top', 'auto');
    }
}

//    Adjust width of report buttons
function adjustReportWidth() {
    var button_length = 0;
    var report_num = 0
    $("#report").find('label').first().css('border-left', '0px');
    $("#report").find('label').each(function() {
        if ($(this).is(':visible')) {
            button_length += $(this).width();
            report_num += 1;
            $(this).css({'border-top': '0px'});
        }
    });
    length_to_add = ($("#report").width() - button_length) / report_num;
    $('#report').find("label").each(function() {$(this).css("width", "+=" + (length_to_add));});
}
function adjustBottomBorder() {
    $('#report').find('label').each(function() {
        if ($(this).hasClass('ui-state-active')) {
            $(this).css('border-bottom-width', '0px');
        } else {
            $(this).css('border-bottom', '1px solid #D3D3D3');
        }
    });
}

function autoCompleteWord(db_url) {
    $("#q").autocomplete({
        source: db_url + "/scripts/term_list.py",
        minLength: 2,
        "dataType": "json",
        focus: function( event, ui ) {
            q = ui.item.label.replace(/<\/?span[^>]*?>/g, '');
            //$("#" + field).val(q); This is too sensitive, so disabled
            return false;
        },
        select: function( event, ui ) {
            q = ui.item.label.replace(/<\/?span[^>]*?>/g, '');
            $("#q").val(q);
            return false;
        }
    }).data("ui-autocomplete")._renderItem = function (ul, item) {
        term = item.label.replace(/^[^<]*/g, '');
        return $("<li></li>")
            .data("item.autocomplete", item)
            .append("<a>" + term + "</a>")
            .appendTo(ul);
    };
}

function autoCompleteMetadata(metadata, field, db_url) {
    $("#" + field).autocomplete({
        source: db_url + "/scripts/metadata_list.py?field=" + field,
        minLength: 2,
        timeout: 1000,
        dataType: "json",
        focus: function( event, ui ) {
            q = ui.item.label.replace(/<\/?span[^>]*?>/g, '');
            q = q.replace(/ CUTHERE /, ' ');
            //$("#" + field).val(q); This is too sensitive, so disabled
            return false;
        },
        select: function( event, ui ) {
            q = ui.item.label.replace(/<\/?span[^>]*?>/g, '');
            q = q.replace(/ CUTHERE /, ' ');
            q = q.split('|');
            q[q.length-1] = '\"' + q[q.length-1].replace(/^\s*/g, '') + '\"'; 
            q = q.join('|');
            $("#" + field).val(q);
            return false;
        }
    }).data("ui-autocomplete")._renderItem = function (ul, item) {
        term = item.label.replace(/.*(?=CUTHERE)CUTHERE /, '');
        return $("<li></li>")
            .data("item.autocomplete", item)
            .append("<a>" + term + "</a>")
            .appendTo(ul);
     };
}

// Display different search parameters based on the type of report used
function showHide(value) {
    $("#results_per_page, #collocation_num, #time_series_num, #year_interval, #method, #metadata_fields").hide();
    $('[id^="explain_"]').hide();
    $("[id$='_question']").hide();
    $('#bottom_search').hide()
    if (value == 'collocation') {
        $("#collocation_num, #metadata_fields").show();
        $('#metadata_fields').find('tr').has('#date').show();
        $('#metadata_fields').find('tr').has('#create_date').show();
        $('#colloc_question').show();
        $('#search_terms_container').slideDown(300);
    }
    if (value == 'kwic' || value == "concordance") {
        $("#results_per_page, #method, #metadata_fields").show();
        $('#metadata_fields').find('tr').has('#date').show();
        $('#metadata_fields').find('tr').has('#create_date').show();
        $('#' + value + '_question').show();
        $('#start_date, #end_date').val('');
        $('#search_terms_container').slideDown(300);
    }
    if (value == 'relevance') {
        $("#results_per_page").show();
        $('#relev_question').show();
    }
    if (value == "time_series") {
        $("#time_series_num, #year_interval, #method, #metadata_fields").show();
        $('#metadata_fields').find('tr').has('#date').hide();
        $('#metadata_fields').find('tr').has('#create_date').hide(); // Temporary hack for Frantext
        $('#time_question').show();
        $('#date').val('');
        $('#create_date').val('');  // Temporary hack for Frantext
        $('#search_terms_container').slideDown(300);
    }
    if (value == "frequencies") {
        $('#search_terms_container, #search_explain, #method, #results_per_page').hide();
        $('#metadata_fields, #bottom_search').show();   
    }
    adjustBottomBorder();
}

//  Function to show or hide search options
function showMoreOptions(display) {
    $("#more_options").button('option', 'label', 'Hide search options');
    if (display == "all") {
        var report = $('input[name=report]:checked', '#search').val();
        showHide(report);
        $("#search_elements").css({'max-height': '614px', 'opacity': 100});
    }
    $('#form_body').css('z-index', 99);
    $("body").append("<div id='search_overlay'></div>");
    var header_height = $('#header').height();
    var footer_height = $('#footer').height();
    var overlay_height = $(document).height() - header_height - footer_height;
    $("#search_overlay")
    .height(overlay_height)
    .css({
       'opacity' : 0.2,
       'margin-top': header_height,
     });
    //$("#search_overlay").fadeIn('fast');
    $("#search_overlay, #header, #footer").click(function() {
        hideSearchForm();
    });
}

function hideSearchForm() {
    $('#search_explain').accordion({
        collapsible: true,
        heightStyle: "content",
        active: false
    });
    $("#search_elements").css({'max-height': 0, 'opacity': 0});
    $("#search_overlay").css('opacity', 0);
    setTimeout(function() {$("#search_overlay").remove();}, 300);
    $("#more_options").button('option', 'label', 'Show search options');
    $('#search_explain').fadeOut(300);
}

(function($)
{
    $.fn.blink = function(options) {
        var defaults = { delay:1500 };
        var options = $.extend(defaults, options);

        return this.each(function() {
            var obj = $(this);
            var state = false;
            var colorchange = setInterval(function() {
                if(state)
                {
                    $(obj).animate({color: '#555555 !important'}, 1500);
                    state = false;
                }
                else
                {
                    $(obj).animate({color: 'rgb(255,0,0) !important'}, 1500);
                    state = true;
                }
            }, options.delay);
            $('#reset_form, #reset_form1').click(function() {
                $(obj).css('color', '#555555 !important');
                clearInterval(colorchange);
            });
        });
    }
}(jQuery))