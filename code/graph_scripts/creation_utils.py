import pandas as pd
import tldextract
from .cookies import *
import pymysql
import sqlite3
import ast
from urllib.parse import urlparse


def read_tables(conn, visit_id):

     
    df_http_requests = pd.read_sql_query("SELECT visit_id, request_id, "
                                         "url, headers, top_level_url, resource_type, "
                                         f"time_stamp, post_body, post_body_raw from http_requests where {visit_id} = visit_id", conn)
    df_http_responses = pd.read_sql_query("SELECT visit_id, request_id, "
                                          "url, headers, response_status, time_stamp, content_hash "
                                          f" from http_responses where {visit_id} = visit_id", conn)
    df_http_redirects = pd.read_sql_query("SELECT visit_id, old_request_id, "
                                          "old_request_url, new_request_url, response_status, "
                                          f"headers, time_stamp from http_redirects where {visit_id} = visit_id", conn)
    call_stacks = pd.read_sql_query(
        f"SELECT visit_id, request_id, call_stack from callstacks where {visit_id} = visit_id", conn)
    javascript = pd.read_sql_query("SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, document_url, symbol, call_stack, operation,"
                                   f" arguments, attributes, value, time_stamp from javascript where {visit_id} = visit_id", conn)
    return df_http_requests, df_http_responses, df_http_redirects, call_stacks, javascript

"""
def get_last_redirect_request_id(conn, visit_id):
    query = f"SELECT referrer, request_id FROM http_requests WHERE visit_id = {visit_id} AND referrer != '' AND referrer IS NOT NULL ORDER BY request_id ASC"
    rows = pd.read_sql_query(query, conn)

    # Function to extract domain from URL
    def extract_domain(url):
        try:
            u = tldextract.extract(url)
            return u.domain+"."+u.suffix
        except:
            return None

    # Process the rows
    last_domain = None
    consecutive_count = 0
    first_request_id = None  # Variable to store the first request_id of the current domain

    for index, row in rows.iterrows():
        referrer = row['referrer']
        request_id = row['request_id']
        domain = extract_domain(referrer)

        if domain:
            if domain == last_domain:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_domain = domain
                first_request_id = request_id  # Update the first_request_id whenever a new domain is encountered

            if consecutive_count == 15:
                print(f"Domain: {domain}, Request ID: {first_request_id}")
                break
    return first_request_id
"""
def get_last_redirect_request_id(conn, visit_id):
    first_request_id = 0

    # Function to extract domain from URL
    def extract_domain(url):
        try:
            u = tldextract.extract(url)
            return u.domain + "." + u.suffix
        except:
            return None

    query = f"SELECT top_level_url FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id DESC LIMIT 1"
    result = pd.read_sql_query(query, conn)
    last_url = result['top_level_url'].iloc[0]

    domain = extract_domain(last_url)

    # Find the first row with the same domain
    query2 = f"SELECT request_id, top_level_url FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id ASC"
    result2 = pd.read_sql_query(query2, conn)
    for index, row in result2.iterrows():
        if extract_domain(row['top_level_url']) == domain:
            # print("Domain: " + domain)
            # print("\tPhase A end with request ID: ", row['request_id'])
            first_request_id = row['request_id']        
            break
    return first_request_id

def get_last_redirect_timestamp(conn, visit_id):
    first_time_stamp = ""

    # Function to extract domain from URL
    def extract_domain(url):
        try:
            u = tldextract.extract(url)
            return u.domain + "." + u.suffix
        except:
            return None
    query = f"SELECT top_level_url FROM javascript WHERE visit_id = {visit_id} ORDER BY time_stamp DESC LIMIT 1"
    result = pd.read_sql_query(query, conn)
    last_url = result['top_level_url'].iloc[0]

    domain = extract_domain(last_url)

    # Find the first row with the same domain
    query2 = f"SELECT time_stamp, top_level_url FROM javascript WHERE visit_id = {visit_id}  ORDER BY time_stamp ASC"
    result2 = pd.read_sql_query(query2, conn)
    for index, row in result2.iterrows():
        if extract_domain(row['top_level_url']) == domain:
            # print("Domain: " + domain)
            # print("\tPhase A end with request ID: ", row['request_id'])
            first_time_stamp = row['time_stamp']
            break
    return first_time_stamp

def get_time_cutoff(conn, visit_id, request_id):

    # Define a function to get the latest time_stamp for a given request_id in a table
    def get_latest_timestamp(table_name, visit_id, request_id):

        if table_name == "http_redirects":
            query = f"SELECT MAX(time_stamp) as latest_time_stamp FROM {table_name} WHERE visit_id = {visit_id} AND CAST(old_request_id AS INT) <= {request_id}"
            result = pd.read_sql_query(query, conn)
            return result['latest_time_stamp'].iloc[0]
        else:
            query = f"SELECT MAX(time_stamp) as latest_time_stamp FROM {table_name} WHERE visit_id = {visit_id} AND CAST(request_id AS INT) <= {request_id}"
            result = pd.read_sql_query(query, conn)
            return result['latest_time_stamp'].iloc[0]

    # Get the latest time_stamp from each table
    latest_timestamp_requests = get_latest_timestamp("http_requests", visit_id, request_id)
    latest_timestamp_redirects = get_latest_timestamp("http_redirects", visit_id, request_id)
    latest_timestamp_responses = get_latest_timestamp("http_responses", visit_id, request_id)

    # Determine the latest time_stamp among the three
    if latest_timestamp_redirects == None:
        # print("======== Redirect table is empty =========")
        latest_timestamp = max(latest_timestamp_requests, latest_timestamp_responses)
    else:
        latest_timestamp = max(latest_timestamp_requests, latest_timestamp_redirects, latest_timestamp_responses)
    # print("\tPhase A end with time_stamp:", latest_timestamp)
    return latest_timestamp


def read_tables_phase1(conn, visit_id):

    request_id = get_last_redirect_request_id(conn, visit_id)
    time_stamp = get_last_redirect_timestamp(conn, visit_id)

    # Javascript table has no request_id. So find out the latest time_stamp value responding to the request id
    # time_cutoff = get_time_cutoff(conn, visit_id, request_id)
    # time_cutoff = "2023-11-13T20:46:34.889Z"

    # Reading and filtering other tables by timestamp
    df_http_requests_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, url, headers, top_level_url, "
                                         f"resource_type, time_stamp, post_body, post_body_raw "
                                         f"FROM http_requests WHERE visit_id = {visit_id} AND CAST(request_id AS INT) <= {request_id}", conn)
    df_http_responses_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, url, headers, response_status, "
                                          f"time_stamp, content_hash FROM http_responses WHERE visit_id = {visit_id} AND CAST(request_id AS INT) <= {request_id}", conn)
    df_http_redirects_phase1 = pd.read_sql_query(
        f"SELECT visit_id, old_request_id, old_request_url, new_request_url, "
                                          f"response_status, headers, time_stamp FROM http_redirects WHERE visit_id = {visit_id} AND CAST(old_request_id AS INT) <= {request_id}", conn)

    javascript_phase1 = pd.read_sql_query(
        f"SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, "
        f"document_url, symbol, call_stack, operation, arguments, attributes, value, time_stamp "
        f"FROM javascript WHERE visit_id = {visit_id}", conn)


    javascript_phase1['time_stamp'] = pd.to_datetime(javascript_phase1['time_stamp'])
    time_stamp_datetime = pd.to_datetime(time_stamp)

    javascript_phase1 = javascript_phase1[javascript_phase1['time_stamp'] <= time_stamp_datetime]
    
    #javascript_cookies_phase1 = pd.read_sql_query(
    #    f"SELECT * FROM javascript_cookies WHERE visit_id = {visit_id} AND time_stamp < '{time_cutoff}'", conn)

    call_stacks_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, call_stack from callstacks where {visit_id} = visit_id AND CAST(request_id AS INT) <= {request_id}", conn)

    return df_http_requests_phase1, df_http_responses_phase1, df_http_redirects_phase1, call_stacks_phase1, javascript_phase1

def add_marker_column(
        df_requests, df_responses, df_redirects, call_stacks, javascript,
        df_requests_phase1, df_responses_phase1, df_redirects_phase1, call_stacks_phase1, javascript_phase1
        ):
    # Create a boolean mask to mark rows that are also in df_http_requests_phase1
    if not df_requests_phase1.empty:
        df_requests['is_in_phase1'] = df_requests['request_id'].isin(df_requests_phase1['request_id'])
    else:
        df_requests['is_in_phase1'] = [False] * len(df_requests)
    if not df_responses_phase1.empty:
        df_responses['is_in_phase1'] = df_responses['request_id'].isin(df_responses_phase1['request_id'])
    else:
        df_responses['is_in_phase1'] = [False] * len(df_responses)
    if not df_redirects_phase1.empty:
        df_redirects['is_in_phase1'] = df_redirects['old_request_id'].isin(df_redirects_phase1['old_request_id'])
    else:
        df_redirects['is_in_phase1'] = [False] * len(df_redirects)
    if not javascript_phase1.empty:
        javascript['is_in_phase1'] = javascript['time_stamp'].isin(javascript_phase1['time_stamp'])
    else:
        javascript['is_in_phase1'] = [False] * len(javascript)
    if not call_stacks_phase1.empty:
        call_stacks['is_in_phase1'] = call_stacks['request_id'].isin(call_stacks_phase1['request_id'])
    else:
        call_stacks['is_in_phase1'] = [False] * len(call_stacks)

    return df_requests, df_responses, df_redirects, call_stacks, javascript
    
def get_sites_visit(conn):

    #df_successful_sites = pd.read_sql_query("SELECT visit_id from crawl_history where "
    #                                        "command = 'GetCommand' and command_status = 'ok'", conn)

    df_successful_sites = pd.read_sql_query("SELECT visit_id from crawl_history where "
                                            "command = 'GetCommand'", conn)
    
    successful_vids = df_successful_sites['visit_id'].tolist()
    print(successful_vids)
    query = "SELECT visit_id, site_url from site_visits where visit_id in %s" % str(tuple(successful_vids))
    
    return pd.read_sql_query(query, conn)

def get_cookies_info(conn, visit_id):

    df_js_cookie = pd.read_sql("select visit_id, time_stamp, script_url, "
                               "document_url, top_level_url, call_stack, operation, "
                               "value from javascript where symbol='window.document.cookie'", conn, parse_dates=['time_stamp'])

    return df_js_cookie

def get_local_db_connection(db_file):

    conn = sqlite3.connect(db_file)
    return conn


def get_remote_db_connection(host_name, user_name, password, ssl_ca, ssl_key, ssl_cert, database_name, port):
    return pymysql.connect(host=host_name, port=port, user=user_name, password=password,
                           database=database_name, ssl_ca=ssl_ca, ssl_key=ssl_key, ssl_cert=ssl_cert)

# Method to read the sqlite file generated by the scraper
def read_scraper_data(db_file):
    dict_scraper = {}
    with  sqlite3.connect(db_file) as con:
        cur = con.cursor()
        for row in cur.execute('SELECT * FROM posts;'):
            title = row[0].strip()
            # Top Level URL
            top_level_url_scraper = ast.literal_eval(row[4])
            # I may need to retrieve more information here when I parse multiple crawls
            filter_ = row[6].strip()
            dict_scraper[title] = (top_level_url_scraper,filter_)
    return dict_scraper

