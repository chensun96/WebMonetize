import pandas as pd
from .cookies import *
import pymysql
import sqlite3
import ast


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

def get_time_cutoff(conn, visit_id):

    # Retrieve the first request_id for the specific visit_id
    first_request_id_query = f"SELECT request_id FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id ASC LIMIT 1"
    first_request_id_df = pd.read_sql_query(first_request_id_query, conn)
    first_request_id = first_request_id_df['request_id'].iloc[0]
    print("first_request_id: ", first_request_id)

    # Define a function to get the latest time_stamp for a given request_id in a table
    def get_latest_timestamp(table_name, request_id, visit_id):
        if table_name == "http_redirects":
            query = f"SELECT MAX(time_stamp) as latest_time_stamp FROM {table_name} WHERE visit_id = {visit_id} AND old_request_id = {request_id}"
            result = pd.read_sql_query(query, conn)
            return result['latest_time_stamp'].iloc[0]
        else:
            query = f"SELECT MAX(time_stamp) as latest_time_stamp FROM {table_name} WHERE visit_id = {visit_id} AND request_id = {request_id}"
            result = pd.read_sql_query(query, conn)
            return result['latest_time_stamp'].iloc[0]

    # Get the latest time_stamp from each table
    latest_timestamp_requests = get_latest_timestamp("http_requests", first_request_id, visit_id)
    latest_timestamp_redirects = get_latest_timestamp("http_redirects", first_request_id, visit_id)
    latest_timestamp_responses = get_latest_timestamp("http_responses", first_request_id, visit_id)

    # Determine the latest time_stamp among the three
    latest_timestamp = max(latest_timestamp_requests, latest_timestamp_redirects, latest_timestamp_responses)
    print("Latest time_stamp for the first request_id:", latest_timestamp)

    return latest_timestamp

def read_tables_phase1(conn, visit_id):   
    
    time_cutoff = get_time_cutoff(conn,visit_id)
    #time_cutoff = "2023-11-13T20:46:34.889Z"

    # Reading and filtering other tables by timestamp
    df_http_requests_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, url, headers, top_level_url, "
                                         f"resource_type, time_stamp, post_body, post_body_raw "
                                         f"FROM http_requests WHERE visit_id = {visit_id} AND time_stamp <= '{time_cutoff}'", conn)
    df_http_responses_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, url, headers, response_status, "
                                          f"time_stamp, content_hash FROM http_responses WHERE visit_id = {visit_id} AND time_stamp <= '{time_cutoff}'", conn)
    df_http_redirects_phase1 = pd.read_sql_query(
        f"SELECT visit_id, old_request_id, old_request_url, new_request_url, "
                                          f"response_status, headers, time_stamp FROM http_redirects WHERE visit_id = {visit_id} AND time_stamp <= '{time_cutoff}'", conn)
    javascript_phase1 = pd.read_sql_query(
        f"SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, "
                                   f"document_url, symbol, call_stack, operation, arguments, attributes, value, time_stamp "
                                   f"FROM javascript WHERE visit_id = {visit_id} AND time_stamp <= '{time_cutoff}'", conn)
    #javascript_cookies_phase1 = pd.read_sql_query(
    #    f"SELECT * FROM javascript_cookies WHERE visit_id = {visit_id} AND time_stamp < '{time_cutoff}'", conn)

    # Read callstacks table without filtering
    call_stacks_phase1 = pd.read_sql_query(f"SELECT * FROM callstacks WHERE visit_id = {visit_id}", conn)

    # Filter callstacks based on the request_ids in df_http_requests
    filtered_request_ids = df_http_requests_phase1['request_id'].unique()
    call_stacks_phase1 = call_stacks_phase1[call_stacks_phase1['request_id'].isin(filtered_request_ids)]

    return df_http_requests_phase1, df_http_responses_phase1, df_http_redirects_phase1, call_stacks_phase1, javascript_phase1

def add_marker_column(
        df_requests, df_responses, df_redirects, call_stacks, javascript,
        df_requests_phase1, df_responses_phase1, df_redirects_phase1, call_stacks_phase1, javascript_phase1
        ):
    # Create a boolean mask to mark rows that are also in df_http_requests_phase1
    if not df_requests_phase1.empty:
        df_requests['is_in_phase1'] = df_requests['time_stamp'].isin(df_requests_phase1['time_stamp'])
    else:
        df_requests['is_in_phase1'] = [False] * len(df_requests)
    if not df_responses_phase1.empty:
        df_responses['is_in_phase1'] = df_responses['time_stamp'].isin(df_responses_phase1['time_stamp'])
    else:
        df_responses['is_in_phase1'] = [False] * len(df_responses)
    if not df_redirects_phase1.empty:
        df_redirects['is_in_phase1'] = df_redirects['time_stamp'].isin(df_redirects_phase1['time_stamp'])
    else:
        df_redirects['is_in_phase1'] = [False] * len(df_redirects)
    if not javascript.empty:
        javascript['is_in_phase1'] = javascript['time_stamp'].isin(javascript_phase1['time_stamp'])
    else:
        javascript['is_in_phase1'] = [False] * len(javascript)
    if not call_stacks.empty:
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
