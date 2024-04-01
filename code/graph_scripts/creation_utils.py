import pandas as pd
import tldextract
from .cookies import *
import pymysql
import sqlite3
import ast
import traceback
from requests.packages.urllib3.util.retry import Retry
from urllib.parse import urlparse
import csv
from requests.adapters import HTTPAdapter
import os
from urllib.parse import urlparse, parse_qs, urlunparse
import json
from adblockparser import AdblockRules
import tld
import requests
from tld import get_fld


def extract_domain(url):
        try:
            u = tldextract.extract(url)
            return u.domain + "." + u.suffix
        except:
            return None


def read_file_newline_stripped(fname):
    try:
        with open(fname) as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
        return lines
    except:
        return []


def setupEasyList(easylist_dir):
	# 'easylist': 'https://easylist.to/easylist/easylist.txt' (accessed on July 03, 2023)
	filepath = os.path.join(easylist_dir, "easylist.txt")
	try:
		with open(filepath) as f:
			rules = f.readlines()
			rules = [x.strip() for x in rules]
		f.close()
	except:
		print(
			f"\n[ERROR] setupEasyList()::AdCollector: {str(traceback.format_exc())}\nException occured while reading filter_rules for domain: {self.site}")
		rules = []

	rule_dict = {}
	rule_dict['script'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
									   supported_options=['script', 'domain', 'subdocument'],
									   skip_unsupported_rules=False)
	rule_dict['script_third'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
											 supported_options=['third-party', 'script', 'domain', 'subdocument'],
											 skip_unsupported_rules=False)
	rule_dict['image'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
									  supported_options=['image', 'domain', 'subdocument'],
									  skip_unsupported_rules=False)
	rule_dict['image_third'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
											supported_options=['third-party', 'image', 'domain', 'subdocument'],
											skip_unsupported_rules=False)
	rule_dict['css'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
									supported_options=['stylesheet', 'domain', 'subdocument'],
									skip_unsupported_rules=False)
	rule_dict['css_third'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
										  supported_options=['third-party', 'stylesheet', 'domain', 'subdocument'],
										  skip_unsupported_rules=False)
	rule_dict['xmlhttp'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
										supported_options=['xmlhttprequest', 'domain', 'subdocument'],
										skip_unsupported_rules=False)
	rule_dict['xmlhttp_third'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
											  supported_options=['third-party', 'xmlhttprequest', 'domain',
																 'subdocument'], skip_unsupported_rules=False)
	rule_dict['third'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
									  supported_options=['third-party', 'domain', 'subdocument'],
									  skip_unsupported_rules=False)
	rule_dict['domain'] = AdblockRules(rules, use_re2=False, max_mem=1024 * 1024 * 1024,
									   supported_options=['domain', 'subdocument'], skip_unsupported_rules=False)
	return rule_dict


def matchURL(domain_top_level, current_domain, current_url, resource_type):
    easylist_dir = os.path.abspath("../code/filterlists_new")
    
    rules_dict = setupEasyList(easylist_dir)
    try:
        if domain_top_level == current_domain:
            third_party_check = False
        else:
            third_party_check = True
        if resource_type == 'sub_frame':
            subdocument_check = True
        else:
            subdocument_check = False
        if resource_type == 'script':
            if third_party_check:
                rules = rules_dict['script_third']
                options = {'third-party': True, 'script': True, 'domain': domain_top_level,
                            'subdocument': subdocument_check}
            else:
                rules = rules_dict['script']
                options = {'script': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
        elif resource_type == 'image' or resource_type == 'imageset':
            if third_party_check:
                rules = rules_dict['image_third']
                options = {'third-party': True, 'image': True, 'domain': domain_top_level,
                            'subdocument': subdocument_check}
            else:
                rules = rules_dict['image']
                options = {'image': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
        elif resource_type == 'stylesheet':
            if third_party_check:
                rules = rules_dict['css_third']
                options = {'third-party': True, 'stylesheet': True, 'domain': domain_top_level,
                            'subdocument': subdocument_check}
            else:
                rules = rules_dict['css']
                options = {'stylesheet': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
        elif resource_type == 'xmlhttprequest':
            if third_party_check:
                rules = rules_dict['xmlhttp_third']
                options = {'third-party': True, 'xmlhttprequest': True, 'domain': domain_top_level,
                            'subdocument': subdocument_check}
            else:
                rules = rules_dict['xmlhttp']
                options = {'xmlhttprequest': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
        elif third_party_check:
            rules = rules_dict['third']
            options = {'third-party': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
        else:
            rules = rules_dict['domain']
            options = {'domain': domain_top_level, 'subdocument': subdocument_check}
        return rules.should_block(current_url, options)
    except Exception as e:
        print("error: ", e)
        return False


def read_url_classification(crawl_agent_id, mount_dir):
	classification_dir = os.path.join(mount_dir, "adurl-classification")
	if not (os.path.exists(classification_dir)):
		os.makedirs(classification_dir)
		return None
	classification_file = os.path.join(classification_dir, f"crawl-{crawl_agent_id}.csv")
	if not (os.path.exists(classification_file)):
		return None
	df = pd.read_csv(classification_file)
	return df


def write_url_classification(crawl_agent_id, mount_dir, url, label):
	classification_file = os.path.join(mount_dir, "adurl-classification", f"crawl-{crawl_agent_id}.csv")
	if not (os.path.exists(classification_file)):
		f = open(classification_file, 'w')
		writer = csv.writer(f)
		header = ["url", "label"]
		writer.writerow(header)
		f.close()
	f = open(classification_file, 'a+')
	writer = csv.writer(f)
	row = [url, label]
	writer.writerow(row)
	f.close()
	return


def labelData(domain_url, script_url):
	'''
	# top_domain = domain of the website being visited
	# script_domain = domain of the iframe url
	# script_url = full url of the iframe to be classified
	# resource_type = subframe, image, script
	'''

	top_domain = domain_url
	data_label = False
	for resource_type in ["sub_frame", "image", "script"]:
		try:
			fld = get_fld(script_url)
		except Exception as e:
		#	self.ad_url_classifocation[script_url] = False
			return False
		list_label = matchURL(top_domain, fld, script_url, resource_type)
		data_label = data_label | list_label
		if data_label == True:
			break
	#self.ad_url_classifocation[script_url] = data_label
	# print(script_url, data_label)
	return data_label

def is_url_similar_to_any(url, urls):
    parsed_url1 = urlparse(url)
    query_params1 = parse_qs(parsed_url1.query)
    
    if 'clkt' in query_params1 or "nm" in query_params1:
        query_params1.pop('clkt', None) 
        query_params1.pop('nm', None) 


        for compare_url in urls:
            parsed_url2 = urlparse(compare_url)
            query_params2 = parse_qs(parsed_url2.query)
            if 'clkt' not in query_params2 and "nm" not in query_params1:
                continue
           
            query_params2.pop('clkt', None)
            query_params2.pop('nm', None)
            

            if query_params1 == query_params2:
                #print(f"{url} is similar to {compare_url}")
                return True
    
    return False


def unique_non_aff_tab_ids(conn, visit_id, df_non_aff):

    first_urls = {}
    unique_groups = []
    visited_final_urls = []
    # Find the parent site url
    query = f"SELECT request_id, url FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id ASC LIMIT 1"
    result = pd.read_sql_query(query, conn)
    if result.empty or 'url' not in result:
        # Handle the case where the result is empty
        print("\nNo data found for the given visit_id. Continue")
        return [], first_urls
    
    domain_url = result['url'].iloc[0]
    print("\nDomain_url: ", domain_url)

    df_http_requests_each_tab = pd.read_sql_query("SELECT visit_id, url, tab_id, request_id, "
                                                    "headers, top_level_url, resource_type, "
                                                    f"time_stamp, post_body, post_body_raw from http_requests where visit_id = {visit_id}", conn)
    
    unique_tab_ids_count = df_http_requests_each_tab['tab_id'].nunique()

    if unique_tab_ids_count ==  1:
        print(f"\tBut no non-affiliated links found in this {visit_id}")
        return [], first_urls
    
    else:
        groups = df_http_requests_each_tab.groupby(['visit_id', 'tab_id'])
        for (visit_id, tab_id), group_data in groups:
            if tab_id == -1:
                continue

            print(f"\tVisit ID: {visit_id}, Tab ID: {tab_id}")
            group_data_sorted = group_data.sort_values(by='request_id')
            first_row_url = group_data_sorted.iloc[0]['url']
            

            # if first_row_url found in the df_affiliate
            df_first_row = df_non_aff[df_non_aff['url'] == first_row_url]

            if len(df_first_row) != 0:
                # deduplicate groups based on the uniqueness of the first row's URL
                final_url = get_final_page_url_for_ads(conn, visit_id, tab_id)

                if first_row_url in first_urls.values() or final_url in visited_final_urls:
                    print("\t\tDuplicate link, ignore")

                else:
                    first_urls[(visit_id, tab_id)] = first_row_url
                    unique_groups.append((visit_id, tab_id, group_data_sorted))
                    visited_final_urls.append(final_url)
                    print("\t\tURL is non-affiliated unique URL, include it")

    print(f"\tNumber of non-affiliated links in {domain_url} is {len(unique_groups)}")
    unique_non_aff_tab_ids = [tab_id for _, tab_id, _ in unique_groups]
    #print(unique_ad_tab_ids)
    return unique_non_aff_tab_ids,  first_urls


def unique_aff_tab_ids(conn, visit_id, df_affiliate):
    first_urls = {}
    unique_groups = []
    visited_final_urls = []
    # Find the parent site url
    query = f"SELECT request_id, url FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id ASC LIMIT 1"
    result = pd.read_sql_query(query, conn)
    if result.empty or 'url' not in result:
        # Handle the case where the result is empty
        print("\nNo data found for the given visit_id. Continue")
        return [], first_urls
    
    domain_url = result['url'].iloc[0]
    print("\nDomain_url: ", domain_url)

    df_http_requests_each_tab = pd.read_sql_query("SELECT visit_id, url, tab_id, request_id, "
                                                    "headers, top_level_url, resource_type, "
                                                    f"time_stamp, post_body, post_body_raw from http_requests where visit_id = {visit_id}", conn)
    
    unique_tab_ids_count = df_http_requests_each_tab['tab_id'].nunique()

    if unique_tab_ids_count ==  1:
        print(f"\tBut no affiliate found in this {visit_id}")
        return [], first_urls
    
    else:
        groups = df_http_requests_each_tab.groupby(['visit_id', 'tab_id'])
        for (visit_id, tab_id), group_data in groups:
            if tab_id == -1:
                continue

            print(f"\tVisit ID: {visit_id}, Tab ID: {tab_id}")
            group_data_sorted = group_data.sort_values(by='request_id')
            first_row_url = group_data_sorted.iloc[0]['url']

            # if first_row_url found in the df_affiliate
            df_first_row = df_affiliate[df_affiliate['url'] == first_row_url]
            if len(df_first_row) != 0:

                # deduplicate groups based on the uniqueness of the first row's URL
                final_url = get_final_page_url_for_ads(conn, visit_id, tab_id)

                if first_row_url in first_urls.values() or final_url in visited_final_urls:
                    print("\t\tDuplicate link, ignore")

                else:
                    first_urls[(visit_id, tab_id)] = first_row_url
                    unique_groups.append((visit_id, tab_id, group_data_sorted))
                    visited_final_urls.append(final_url)
                    print("\t\tURL is affiliate unique URL, include it")

    print(f"\tNumber of affiliate in {domain_url} is {len(unique_groups)}")
    unique_aff_tab_ids = [tab_id for _, tab_id, _ in unique_groups]
    #print(unique_ad_tab_ids)
    return unique_aff_tab_ids,  first_urls


def unique_ad_tab_ids_fake(conn, visit_id):

    first_urls = {}
    unique_groups = []
    not_ads = []
    visited_final_urls = []

    # Find the parent site url
    query = f"SELECT request_id, url FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id ASC LIMIT 1"
    result = pd.read_sql_query(query, conn)
    if result.empty or 'url' not in result:
        # Handle the case where the result is empty
        print("\nNo data found for the given visit_id. Continue")
        return [], first_urls
    
    domain_url = result['url'].iloc[0]
    print("\nDomain_url: ", domain_url)

    df_http_requests_each_tab = pd.read_sql_query("SELECT visit_id, url, tab_id, request_id, "
                                                    "headers, top_level_url, resource_type, "
                                                    f"time_stamp, post_body, post_body_raw from http_requests where visit_id = {visit_id}", conn)
    
    unique_tab_ids_count = df_http_requests_each_tab['tab_id'].nunique()
    print(f"Number of unique tab_ids: {unique_tab_ids_count}")
    #if unique_tab_ids_count ==  1:  # youtube long redirect
    if unique_tab_ids_count >=  1:  # Instagram long redirect
    
        groups = df_http_requests_each_tab.groupby(['visit_id', 'tab_id'])
        for (visit_id, tab_id), group_data in groups:
            if tab_id == -1:
                continue

            print(f"\tVisit ID: {visit_id}, Tab ID: {tab_id}")
            group_data_sorted = group_data.sort_values(by='request_id')
            first_row_url = group_data_sorted.iloc[0]['url']
            print(first_row_url)
            curr_fld = get_fld(first_row_url)


            # # same domain for ad URLs is ok
            #if first_row_url == domain_url or curr_fld in ["facebook.com", "instagram.com", "twitter.com", "youtube.com", "linkedin.com", "pinterest.com"]:
            #    continue


            # # same domain for ad urls is not ok
            #if first_row_url == domain_url or curr_fld == get_fld(domain_url) or curr_fld in ["facebook.com", "instagram.com", "twitter.com", "youtube.com", "linkedin.com", "pinterest.com"]:
            #    print("\t\tSame domain, ignore")
            #    continue

            
            # deduplicate groups based on the uniqueness of the first row's URL
            final_url = get_final_page_url_for_ads(conn, visit_id, tab_id)

            if first_row_url in first_urls.values() or final_url in visited_final_urls:
                print("\t\tDuplicate link, ignore")
                
            elif first_row_url in not_ads:
                print("\t\tDuplicate not ads link, ignore")

            # ignore url is is_url_similar_to_any return True 
            elif is_url_similar_to_any(first_row_url, first_urls.values()):
                print("\t\tSimilar ads link, ignore")
            # check if the url is ads
            # elif not(labelData(domain_url, first_row_url)):
            #    not_ads.append(first_row_url)
            #    print("\t\tURL is not ad URL, ignore")

            else:
                first_urls[(visit_id, tab_id)] = first_row_url
                unique_groups.append((visit_id, tab_id, group_data_sorted))
                visited_final_urls.append(final_url)
                print("\t\tURL is ad unique URL, include it")

    print(f"\tNumber of ads in {domain_url} is {len(unique_groups)}")
    unique_ad_tab_ids = [tab_id for _, tab_id, _ in unique_groups]
    #print(unique_ad_tab_ids)
    return unique_ad_tab_ids,  first_urls

def unique_ad_tab_ids(conn, visit_id):

    first_urls = {}
    unique_groups = []
    not_ads = []
    visited_final_urls = []

    # Find the parent site url
    query = f"SELECT request_id, url FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id ASC LIMIT 1"
    result = pd.read_sql_query(query, conn)
    if result.empty or 'url' not in result:
        # Handle the case where the result is empty
        print("\nNo data found for the given visit_id. Continue")
        return [], first_urls
    
    domain_url = result['url'].iloc[0]
    print("\nDomain_url: ", domain_url)

    df_http_requests_each_tab = pd.read_sql_query("SELECT visit_id, url, tab_id, request_id, "
                                                    "headers, top_level_url, resource_type, "
                                                    f"time_stamp, post_body, post_body_raw from http_requests where visit_id = {visit_id}", conn)
    
    unique_tab_ids_count = df_http_requests_each_tab['tab_id'].nunique()
    #print(f"Number of unique tab_ids: {unique_tab_ids_count}")
    if unique_tab_ids_count ==  1:
        print(f"\tNo ads in this {visit_id}")
        return [], first_urls
    
    else:
        groups = df_http_requests_each_tab.groupby(['visit_id', 'tab_id'])
        for (visit_id, tab_id), group_data in groups:
            if tab_id == -1:
                continue

            print(f"\tVisit ID: {visit_id}, Tab ID: {tab_id}")
            group_data_sorted = group_data.sort_values(by='request_id')
            first_row_url = group_data_sorted.iloc[0]['url']
            #print(first_row_url)
            curr_fld = get_fld(first_row_url)


            # # same domain for ad URLs is ok
            if first_row_url == domain_url or curr_fld in ["facebook.com", "instagram.com", "twitter.com", "youtube.com", "linkedin.com", "pinterest.com"]:
                continue


            # # same domain for ad urls is not ok
            #if first_row_url == domain_url or curr_fld == get_fld(domain_url) or curr_fld in ["facebook.com", "instagram.com", "twitter.com", "youtube.com", "linkedin.com", "pinterest.com"]:
            #    print("\t\tSame domain, ignore")
            #    continue

            else:
                # deduplicate groups based on the uniqueness of the first row's URL
                final_url = get_final_page_url_for_ads(conn, visit_id, tab_id)

                if first_row_url in first_urls.values() or final_url in visited_final_urls:
                    print("\t\tDuplicate link, ignore")
                 
                elif first_row_url in not_ads:
                    print("\t\tDuplicate not ads link, ignore")

                # ignore url is is_url_similar_to_any return True 
                elif is_url_similar_to_any(first_row_url, first_urls.values()):
                    print("\t\tSimilar ads link, ignore")
                # check if the url is ads
                elif not(labelData(domain_url, first_row_url)):
                    not_ads.append(first_row_url)
                    print("\t\tURL is not ad URL, ignore")

                else:
                    first_urls[(visit_id, tab_id)] = first_row_url
                    unique_groups.append((visit_id, tab_id, group_data_sorted))
                    visited_final_urls.append(final_url)
                    print("\t\tURL is ad unique URL, include it")

    print(f"\tNumber of ads in {domain_url} is {len(unique_groups)}")
    unique_ad_tab_ids = [tab_id for _, tab_id, _ in unique_groups]
    #print(unique_ad_tab_ids)
    return unique_ad_tab_ids,  first_urls

def get_max_min_request_id(df_http_requests, df_http_responses, df_http_redirects):
    # Convert request_id and old_request_id columns to numeric (integer) explicitly
    df_http_requests['request_id'] = pd.to_numeric(df_http_requests['request_id'], errors='coerce')
    df_http_responses['request_id'] = pd.to_numeric(df_http_responses['request_id'], errors='coerce')
    df_http_redirects['old_request_id'] = pd.to_numeric(df_http_redirects['old_request_id'], errors='coerce')

    # Now proceed to find max and min as before
    max_request_id_http_requests = df_http_requests['request_id'].max()
    min_request_id_http_requests = df_http_requests['request_id'].min()

    max_request_id_http_responses = df_http_responses['request_id'].max()
    min_request_id_http_responses = df_http_responses['request_id'].min()

    max_request_id_http_redirects = df_http_redirects['old_request_id'].max()
    min_request_id_http_redirects = df_http_redirects['old_request_id'].min()

    # Print the results
    #print(f"HTTP Requests - Max Request ID: {max_request_id_http_requests}, Min Request ID: {min_request_id_http_requests}")
    #print(f"HTTP Responses - Max Request ID: {max_request_id_http_responses}, Min Request ID: {min_request_id_http_responses}")
    #print(f"HTTP Redirects (Old Request ID) - Max: {max_request_id_http_redirects}, Min: {min_request_id_http_redirects}")

    # Calculate overall max and min request_id
    overall_max_request_id = max(max_request_id_http_requests, max_request_id_http_responses, max_request_id_http_redirects)
    overall_min_request_id = min(min_request_id_http_requests, min_request_id_http_responses, min_request_id_http_redirects)

    return int(overall_max_request_id), int(overall_min_request_id)


def read_tables_for_ads(conn, visit_id, tab_id):

    df_http_requests = pd.read_sql_query("SELECT visit_id, request_id, "
                                         "url, headers, top_level_url, resource_type, "
                                         f"time_stamp, post_body, post_body_raw from http_requests where visit_id = {visit_id} AND tab_id = {tab_id}",
                                         conn)
    df_http_responses = pd.read_sql_query("SELECT visit_id, request_id, "
                                          "url, headers, response_status, time_stamp, content_hash "
                                          f" from http_responses where visit_id = {visit_id} AND tab_id = {tab_id}", conn)
    df_http_redirects = pd.read_sql_query("SELECT visit_id, old_request_id, "
                                          "old_request_url, new_request_url, response_status, "
                                          f"headers, time_stamp from http_redirects where visit_id = {visit_id} AND tab_id = {tab_id}", conn)

    javascript = pd.read_sql_query(
        "SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, document_url, symbol, call_stack, operation,"
        f" arguments, attributes, value, time_stamp from javascript where visit_id = {visit_id} AND tab_id = {tab_id}", conn)

    # call_stack table doesn't have tab_id.
    # Use request_id to define the range of data
    max_request_id, min_request_id = get_max_min_request_id(df_http_requests, df_http_responses, df_http_redirects)

    query = f"""
        SELECT visit_id, request_id, call_stack
        FROM callstacks
        WHERE visit_id = {visit_id} AND CAST(request_id AS INT) BETWEEN {min_request_id} AND {max_request_id}
    """
    call_stacks = pd.read_sql_query(query, conn)
    call_stacks['request_id'] = call_stacks['request_id'].astype(str)

    return df_http_requests, df_http_responses, df_http_redirects, call_stacks, javascript, max_request_id, min_request_id


def read_tables(conn, visit_id):
     
    df_http_requests = pd.read_sql_query("SELECT visit_id, request_id, "
                                         "url, headers, top_level_url, resource_type, "
                                         f"time_stamp, post_body, post_body_raw from http_requests where visit_id = {visit_id}", conn)
    df_http_responses = pd.read_sql_query("SELECT visit_id, request_id, "
                                          "url, headers, response_status, time_stamp, content_hash "
                                          f" from http_responses where visit_id = {visit_id}", conn)
    df_http_redirects = pd.read_sql_query("SELECT visit_id, old_request_id, "
                                          "old_request_url, new_request_url, response_status, "
                                          f"headers, time_stamp from http_redirects where visit_id = {visit_id}", conn)
    call_stacks = pd.read_sql_query(
        f"SELECT visit_id, request_id, call_stack from callstacks where visit_id = {visit_id}", conn)
    javascript = pd.read_sql_query("SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, document_url, symbol, call_stack, operation,"
                                   f" arguments, attributes, value, time_stamp from javascript where visit_id = {visit_id}", conn)
    return df_http_requests, df_http_responses, df_http_redirects, call_stacks, javascript


def get_last_redirect_request_id_for_ads(conn, visit_id, tab_id):
    first_request_id = 0
    # Fetch all top-level URLs for the visit_id and tab_id, ordered by request_id descending
    query = f"""
    SELECT request_id, top_level_url 
    FROM http_requests 
    WHERE visit_id = {visit_id} AND tab_id = {tab_id} 
    ORDER BY request_id DESC
    """
    result = pd.read_sql_query(query, conn)

    if result.empty:
        print("No data found for the given visit_id. Continue")
        return None

    consecutive_count = 1
    last_domain = None

    for index, row in result.iterrows():
        current_domain = extract_domain(row['top_level_url'])
        if last_domain is None:
            last_domain = current_domain
        elif last_domain == current_domain:
            consecutive_count += 1
            if consecutive_count == 5:
                print("\t\tDomain: " + last_domain)
                # Found 10 consecutive URLs with the same domain
                break
        else:
            # Reset the count if a different domain is found
            consecutive_count = 1
            last_domain = current_domain

        # Find the first row with the same domain
        query2 = f"SELECT request_id, top_level_url FROM http_requests WHERE visit_id = {visit_id} AND tab_id = {tab_id} ORDER BY request_id ASC"
        result2 = pd.read_sql_query(query2, conn)
        for index, row in result2.iterrows():
            if extract_domain(row['top_level_url']) == last_domain:
                #print("Domain: " + last_domain)
                #print("\tPhase A end with request ID: ", row['request_id'])
                first_request_id = row['request_id']
                break

    return first_request_id

"""
def get_last_redirect_request_id_for_ads(conn, visit_id, tab_id):
    # TODO: top level url for the last request id could from other domain
    # Change to if last 10 consecutive top level url are the same, then extrain the domain. 
    first_request_id = 0

    query = f"SELECT top_level_url FROM http_requests WHERE visit_id = {visit_id} AND tab_id = {tab_id} ORDER BY request_id DESC LIMIT 1"
    result = pd.read_sql_query(query, conn)

    if result.empty or 'top_level_url' not in result:
        # Handle the case where the result is empty
        print("No data found for the given visit_id. Continue")
        return None
    last_url = result['top_level_url'].iloc[0]

    domain = extract_domain(last_url)

    # Find the first row with the same domain
    query2 = f"SELECT request_id, top_level_url FROM http_requests WHERE visit_id = {visit_id} AND tab_id = {tab_id} ORDER BY request_id ASC"
    result2 = pd.read_sql_query(query2, conn)
    for index, row in result2.iterrows():
        if extract_domain(row['top_level_url']) == domain:
            # print("Domain: " + domain)
            # print("\tPhase A end with request ID: ", row['request_id'])
            first_request_id = row['request_id']
            break
    return first_request_id
"""

def get_last_redirect_timestamp_for_ads(conn, visit_id, tab_id):
    first_time_stamp = ""

    # Function to extract domain from URL

    query = f"SELECT top_level_url FROM javascript WHERE visit_id = {visit_id} AND tab_id = {tab_id} ORDER BY time_stamp DESC LIMIT 1"
    result = pd.read_sql_query(query, conn)

    if result.empty or 'top_level_url' not in result:
        # Handle the case where the result is empty
        print("No data found in javascript table for the given visit_id. Ignore this url")
        return None

    last_url = result['top_level_url'].iloc[0]

    domain = extract_domain(last_url)

    # Find the first row with the same domain
    query2 = f"SELECT time_stamp, top_level_url FROM javascript WHERE visit_id = {visit_id} AND tab_id = {tab_id} ORDER BY time_stamp ASC"
    result2 = pd.read_sql_query(query2, conn)
    for index, row in result2.iterrows():
        if extract_domain(row['top_level_url']) == domain:
            # print("Domain: " + domain)
            # print("\tPhase A end with request ID: ", row['request_id'])
            first_time_stamp = row['time_stamp']
            break
    return first_time_stamp


def read_tables_phase1_for_ads(conn, visit_id, tab_id, max_request_id, min_request_id):
    request_id = get_last_redirect_request_id_for_ads(conn, visit_id, tab_id)
    time_stamp = get_last_redirect_timestamp_for_ads(conn, visit_id, tab_id)

    # Reading and filtering tables
    df_http_requests_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, url, headers, top_level_url, "
        f"resource_type, time_stamp, post_body, post_body_raw "
        f"FROM http_requests WHERE visit_id = {visit_id} AND tab_id = {tab_id} AND CAST(request_id AS INT) <= {request_id}", conn)
    
    df_http_requests_phase1['request_id'] = df_http_requests_phase1['request_id'].astype(str)

    df_http_responses_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, url, headers, response_status, "
        f"time_stamp, content_hash FROM http_responses WHERE visit_id = {visit_id} AND tab_id = {tab_id} AND CAST(request_id AS INT) <= {request_id}",
        conn)
    
    df_http_responses_phase1['request_id'] = df_http_responses_phase1['request_id'].astype(str)

    df_http_redirects_phase1 = pd.read_sql_query(
        f"SELECT visit_id, old_request_id, old_request_url, new_request_url, "
        f"response_status, headers, time_stamp FROM http_redirects WHERE visit_id = {visit_id} AND tab_id = {tab_id} AND CAST(old_request_id AS INT) <= {request_id}",
        conn)
    
    df_http_redirects_phase1['old_request_id'] = df_http_redirects_phase1['old_request_id'].astype(str)

    javascript_phase1 = pd.read_sql_query(
        f"SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, "
        f"document_url, symbol, call_stack, operation, arguments, attributes, value, time_stamp "
        f"FROM javascript WHERE visit_id = {visit_id} AND tab_id = {tab_id}", conn)
    

    # Convert the 'time_stamp' column to datetime objects
    javascript_phase1['time_stamp'] = pd.to_datetime(javascript_phase1['time_stamp'])

    # Convert the 'time_cutoff' string to a datetime object
    time_stamp_datetime = pd.to_datetime(time_stamp)

    # Filter the DataFrame based on the time cutoff
    javascript_phase1 = javascript_phase1[javascript_phase1['time_stamp'] <= time_stamp_datetime]
    javascript_phase1['time_stamp'] = javascript_phase1['time_stamp'].astype(str)


    #print(javascript_phase1['time_stamp'])

    query = f"""
            SELECT visit_id, request_id, call_stack
            FROM callstacks
            WHERE visit_id = {visit_id} AND CAST(request_id AS INT) BETWEEN {min_request_id} AND {max_request_id} AND CAST(request_id AS INT) <= {request_id}
        """
    call_stacks_phase1 = pd.read_sql_query(query, conn)
    call_stacks_phase1['request_id'] = call_stacks_phase1['request_id'].astype(str)

    return df_http_requests_phase1, df_http_responses_phase1, df_http_redirects_phase1, call_stacks_phase1, javascript_phase1


"""
def get_last_redirect_request_id(conn, visit_id):
    query = f"SELECT referrer, request_id FROM http_requests WHERE visit_id = {visit_id} AND referrer != '' AND referrer IS NOT NULL ORDER BY request_id ASC"
    rows = pd.read_sql_query(query, conn)

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

    query = f"SELECT top_level_url FROM http_requests WHERE visit_id = {visit_id} ORDER BY request_id DESC LIMIT 1"
    result = pd.read_sql_query(query, conn)

    if result.empty or 'top_level_url' not in result:
        # Handle the case where the result is empty
        print("No data found for the given visit_id. Continue")
        return None
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
    
    query = f"SELECT top_level_url FROM javascript WHERE visit_id = {visit_id} ORDER BY time_stamp DESC LIMIT 1"
    result = pd.read_sql_query(query, conn)

    if result.empty or 'top_level_url' not in result:
        # Handle the case where the result is empty
        print("No data found in javascript table for the given visit_id. Continue")
        return None
    
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
    df_http_requests_phase1['request_id'] = df_http_requests_phase1['request_id'].astype(str)

    df_http_responses_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, url, headers, response_status, "
                                          f"time_stamp, content_hash FROM http_responses WHERE visit_id = {visit_id} AND CAST(request_id AS INT) <= {request_id}", conn)
    df_http_responses_phase1['request_id'] = df_http_responses_phase1['request_id'].astype(str)

    df_http_redirects_phase1 = pd.read_sql_query(
        f"SELECT visit_id, old_request_id, old_request_url, new_request_url, "
                                          f"response_status, headers, time_stamp FROM http_redirects WHERE visit_id = {visit_id} AND CAST(old_request_id AS INT) <= {request_id}", conn)
    df_http_redirects_phase1['old_request_id'] = df_http_redirects_phase1['old_request_id'].astype(str)

    javascript_phase1 = pd.read_sql_query(
        f"SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, "
        f"document_url, symbol, call_stack, operation, arguments, attributes, value, time_stamp "
        f"FROM javascript WHERE visit_id = {visit_id}", conn)


    javascript_phase1['time_stamp'] = pd.to_datetime(javascript_phase1['time_stamp'])
    time_stamp_datetime = pd.to_datetime(time_stamp)

    javascript_phase1 = javascript_phase1[javascript_phase1['time_stamp'] <= time_stamp_datetime]
    javascript_phase1['time_stamp'] = javascript_phase1['time_stamp'].astype(str)
    
    #javascript_cookies_phase1 = pd.read_sql_query(
    #    f"SELECT * FROM javascript_cookies WHERE visit_id = {visit_id} AND time_stamp < '{time_cutoff}'", conn)

    call_stacks_phase1 = pd.read_sql_query(
        f"SELECT visit_id, request_id, call_stack from callstacks where visit_id = {visit_id} AND CAST(request_id AS INT) <= {request_id}", conn)
    call_stacks_phase1['request_id'] = call_stacks_phase1['request_id'].astype(str)
    
    return df_http_requests_phase1, df_http_responses_phase1, df_http_redirects_phase1, call_stacks_phase1, javascript_phase1


def get_final_page_url(conn, visit_id):
    request_id = get_last_redirect_request_id(conn, visit_id)
    query = f"""
    SELECT url, time_stamp 
    FROM http_requests 
    WHERE visit_id = {visit_id} AND CAST(request_id AS INT) = {request_id} 
    ORDER BY time_stamp DESC 
    LIMIT 1
    """
    df_url = pd.read_sql_query(query, conn)
    return df_url.iloc[0]['url']


def get_final_page_url_for_ads(conn, visit_id, tab_id):
    request_id = get_last_redirect_request_id_for_ads(conn, visit_id, tab_id)
    query = f"""
    SELECT url, time_stamp 
    FROM http_requests 
    WHERE CAST(visit_id AS INT) = {visit_id} AND tab_id = {tab_id} AND CAST(request_id AS INT) = {request_id} 
    ORDER BY time_stamp DESC 
    LIMIT 1
    """
    df_url = pd.read_sql_query(query, conn)
    return df_url.iloc[0]['url']


def add_marker_column(
        df_requests, df_responses, df_redirects, call_stacks, javascript,
        df_requests_phase1, df_responses_phase1, df_redirects_phase1, call_stacks_phase1, javascript_phase1
        ):
    # Create a boolean mask to mark rows that are also in df_http_requests_phase1
    #print("df_requests request_id: ", df_requests['request_id'].dtype)
    #print("df_responses request_id: ", df_responses['request_id'].dtype)
    #print("df_redirects request_id: ", df_redirects['old_request_id'].dtype)
    #print("call_stacks request_id: ", call_stacks['request_id'].dtype)
    #print("javascript time_stamp: ", javascript['time_stamp'].dtype)


  
    #print("df_requests_phase1 request_id: ", df_requests_phase1['request_id'].dtype)
    #print("df_responses_phase1 request_id: ", df_responses_phase1['request_id'].dtype)
    #print("df_redirects_phase1 request_id: ", df_redirects_phase1['old_request_id'].dtype)
    #print("call_stacks_phase1 request_id: ", call_stacks_phase1['request_id'].dtype)
    #print("javascript_phase1 time_stamp: ", javascript_phase1['time_stamp'].dtype)

    if not df_requests_phase1.empty:
        df_requests['is_in_phase1'] = df_requests['request_id'].isin(df_requests_phase1['request_id'])
        #print("len of true in df_requests: ", df_requests[df_requests['is_in_phase1']].shape[0])
    else:
        df_requests['is_in_phase1'] = [False] * len(df_requests)

    if not df_responses_phase1.empty:
        df_responses['is_in_phase1'] = df_responses['request_id'].isin(df_responses_phase1['request_id'])
        #print("len of true in df_responses: ", df_responses[df_responses['is_in_phase1']].shape[0])
    else:
        df_responses['is_in_phase1'] = [False] * len(df_responses)

    if not df_redirects_phase1.empty:
        df_redirects['is_in_phase1'] = df_redirects['old_request_id'].isin(df_redirects_phase1['old_request_id'])
        #print("len of true in df_redirects: ", df_redirects[df_redirects['is_in_phase1']].shape[0])
    else:
        df_redirects['is_in_phase1'] = [False] * len(df_redirects)

    if not javascript_phase1.empty:
        javascript['is_in_phase1'] = javascript['time_stamp'].isin(javascript_phase1['time_stamp'])
        #print("len of true in javascript: ",javascript[javascript['is_in_phase1']].shape[0])
    else:
        javascript['is_in_phase1'] = [False] * len(javascript)

    if not call_stacks_phase1.empty:
        call_stacks['is_in_phase1'] = call_stacks['request_id'].isin(call_stacks_phase1['request_id'])
        #print("len of true in call_stacks: ",call_stacks[call_stacks['is_in_phase1']].shape[0])
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

    if len(successful_vids) == 1:
        # If there's only one visit_id, format it without a tuple
        query = "SELECT visit_id, site_url FROM site_visits WHERE visit_id = %s" % successful_vids[0]
    else:
        # If there are multiple visit_ids, format it using a tuple
        query = "SELECT visit_id, site_url FROM site_visits WHERE visit_id IN %s" % str(tuple(successful_vids))

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

