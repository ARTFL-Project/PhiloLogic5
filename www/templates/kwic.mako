<%include file="header.mako"/>
<%include file="search_form.mako"/>
<div class="container-fluid" id='philologic_response'>
    <div id='initial_report'>
       <div id='description'>
            <%
             start, end, n = f.link.page_interval(results_per_page, results, q["start"], q["end"])
             r_status = "."
             if not results.done:
                r_status += " Still working..."
             current_pos = start
            %>
            <div id="search_arguments">
                Searching database for <b>${q['q'].decode('utf-8', 'ignore')}</b></br>
                Bibliographic criteria: ${biblio_criteria or "None"}
            </div>
            % if end != 0:
                % if end < results_per_page or end < len(results):
                    Hits <span id="start">${start}</span> - <span id="end">${end}</span> of <span id="total_results">${len(results) or results_per_page}</span><span id="incomplete">${r_status}</span>
                % else:
                    Hits <span id="start">${start}</span> - <span id="end">${len(results) or end}</span> of <span id="total_results">${len(results) or results_per_page}</span><span id="incomplete">${r_status}</span>         
                % endif
            % else:
                No results for your query.
            % endif
        </div>
    </div>
    <div class="row" id="act-on-report">
        <div class="col-xs-12 col-md-6">
            <div id="report_switch" class="btn-group" data-toggle="buttons">
                <label class="btn btn-primary">
                    <input type="radio" name="report_switch" id="concordance_switch" value="?${q['q_string'].replace('report=kwic', 'report=concordance')}">
                    View occurences with context
                </label>
                <label class="btn btn-primary active">
                    <input type="radio" name="report_switch" id="kwic_switch" value="?${q['q_string'].replace('report=concordance', 'report=kwic')}" checked=>
                    View occurences line by line (KWIC)
                </label>
            </div>
        </div>
        <div class="col-xs-12 col-md-4 col-md-offset-2">
            <div class="btn-group pull-right">
                <button type="button" class="btn btn-default dropdown-toggle" data-toggle="dropdown">
                    Display frequency by ${config["metadata"][0].title()}<span class="caret"></span>
                </button>
                <ul class="dropdown-menu" role="menu" id="frequency_field">
                    % for facet in config["facets"]:
                        <%
                        if facet in config["metadata_aliases"]:
                            alias = config["metadata_aliases"][facet]
                        else:
                            alias = facet
                        %>
                        <li><a class="sidebar_option" id="side_opt_${facet}" data-value='${facet}' data-display='${facet}'>Display frequency by ${alias}</a></li>
                    % endfor
                    % if report != 'bibliography':
                        <li class="divider"></li>
                        <li><a class="sidebar_option" id="side_opt_collocate" data-value='collocation_report' data-display='collocate'>Display collocates</a></li>
                    % endif
                </ul>
            </div>
        </div>
    </div>
    <div id="results_container" class="results_container">
        <div id="kwic_concordance">
            % for i in fetch_kwic(results, path, q, f.link.byte_query, db, start-1, end):
                <div class="kwic_line">
                    % if len(str(end)) > len(str(current_pos)):
                        <% spaces = ' ' * (len(str(end)) - len(str(current_pos))) %>
                        <span style="white-space:pre-wrap;">${current_pos}.${spaces}</span>
                    % else:
                        <span>${current_pos}.</span>    
                    % endif
                    ${i}
                    <% current_pos += 1 %>
                </div>
            % endfor
        </div>
    </div>
<!--    <div id="results-bibliography">
        <span id="show-results-bibliography">Results Bibliography in current page:</span>
    </div>-->
    <div class="more">
        <%include file="results_paging.mako" args="start=start,results_per_page=results_per_page,q=q,results=results"/> 
        <div style='clear:both;'></div>
    </div>
</div>
<%include file="footer.mako"/>