import argparse
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from adblockparser import AdblockRules
from tld import get_fld
import requests
import os
import graph_scripts as gs
from tqdm import tqdm
import traceback

global_curr_domain = []

def read_sites_visit(db_file, conn):
    """Read the list of sites visited by crawler and their information
    :return: pandas df of site_visits table in scraper SQL file.
    """
    # conn = gs.get_local_db_connection(db_file)
    # Return a dataframe of the sites visited (stored in sites_visits of scraper SQL)
    return gs.get_sites_visit(conn)


def read_sql_crawl_data(visit_id, db_file, conn):
    """Read SQL data from crawler for a given visit_ID.
    :param visit_id: visit ID of a crawl URL.
    :return: Parsed information (nodes and edges) in pandas df.
    """
    # Read tables from DB and store as DataFrames
    df_http_requests, df_http_responses, df_http_redirects, call_stacks, javascript = gs.read_tables(
        conn, visit_id
    )
    return df_http_requests, df_http_responses, df_http_redirects, call_stacks, javascript

def download_lists(FILTERLIST_DIR):
	"""
	Function to download the lists used in AdGraph.
	Args:
		FILTERLIST_DIR: Path of the output directory to which filter lists should be written.
	Returns:
		Nothing, writes the lists to a directory.
	This functions does the following:
	1. Sends HTTP requests for the lists used in AdGraph.
	2. Writes to an output directory.
	"""

	num_retries = 5
	session = requests.Session()
	retry = Retry(total=num_retries, connect=num_retries, read=num_retries, backoff_factor=0.5)
	adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=200)
	session.mount('http://', adapter)
	session.mount('https://', adapter)

	request_headers_https = {
		"Connection": "keep-alive",
		"User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
		"Accept": "*/*",
		"Accept-Encoding": "gzip, deflate, br"
	}
	# "Accept-Language": "en-US,en;q=0.9"

	request_headers_http = {
		"Connection": "keep-alive",
		"User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
		"Accept": "*/*"
	}

	raw_lists = {
		'easylist': 'https://easylist.to/easylist/easylist.txt',
	}
	for listname, url in raw_lists.items():
		with open(os.path.join(FILTERLIST_DIR, "%s.txt" % listname), 'wb') as f:
			# f.write(requests.get(url).content)
			try:
				response = session.get(url, timeout=45, headers=request_headers_https)
				response_content = response.content
				f.write(response_content)
			except requests.exceptions.ConnectionError as e1:
				continue

def read_file_newline_stripped(fname):
	try:
		with open(fname) as f:
			lines = f.readlines()
			lines = [x.strip() for x in lines]
		return lines
	except:
		return []

def setup_filterlists():
	'''
	Setup and download (if not already downloaded earlier) the filter lists to identify ad-related URLs
	'''
	FILTERLIST_DIR = "ads_filterlists"

	if not os.path.isdir(FILTERLIST_DIR):
		os.makedirs(FILTERLIST_DIR)
	download_lists(FILTERLIST_DIR)
	filterlist_rules = {}
	filterlists = os.listdir(FILTERLIST_DIR)

	for fname in filterlists:
		rule_dict = {}
		rules = read_file_newline_stripped(os.path.join(FILTERLIST_DIR, fname))
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
		filterlist_rules[fname] = rule_dict
	return filterlists, filterlist_rules

def match_url(domain_top_level, current_domain, current_url, resource_type, rules_dict):
	'''
	Associate the URL to a particular category based on different rules
	'''
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
				options = {'third-party': True, 'script': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
			else:
				rules = rules_dict['script']
				options = {'script': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
		elif resource_type == 'image' or resource_type == 'imageset':
			if third_party_check:
				rules = rules_dict['image_third']
				options = {'third-party': True, 'image': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
			else:
				rules = rules_dict['image']
				options = {'image': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
		elif resource_type == 'stylesheet':
			if third_party_check:
				rules = rules_dict['css_third']
				options = {'third-party': True, 'stylesheet': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
			else:
				rules = rules_dict['css']
				options = {'stylesheet': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
		elif resource_type == 'xmlhttprequest':
			if third_party_check:
				rules = rules_dict['xmlhttp_third']
				options = {'third-party': True, 'xmlhttprequest': True, 'domain': domain_top_level, 'subdocument': subdocument_check}
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
		return False

def label_data(url):
	'''
	# top_domain = the website being visited
	# script_domain = domain of iframe url
	# script_url = url of iframe
	# resource_type = subframe, image, script
	'''
	data_label = False
	# Skip if the URL is from the same domain or a common domain
	curr_fld = str(get_fld(url)).strip()
	# print("curr_fld: ", curr_fld)
	if (curr_fld in global_curr_domain) or (
			curr_fld in ["facebook.com", "instagram.com", "twitter.com", "youtube.com", "linkedin.com",
						 "pinterest.com"]):
		print("Skip since the URL is from the same domain or a common domain")
		return data_label

	top_domain = global_curr_domain
	filterlists, filterlist_rules = setup_filterlists()
	for fl in filterlists:
		for resource_type in ["sub_frame", "script", "image"]:
			list_label = match_url(top_domain, get_fld(url), url, resource_type, filterlist_rules[fl])
			data_label = data_label | list_label
			if data_label == True:
				print("Found ads: ", url)
				break
		if data_label == True:
			break
	print("Not an ads")
	return data_label



def detect_ads(db_file, ldb_file):
	ads_link = []
	conn = gs.get_local_db_connection(db_file)
	try:
		sites_visits = read_sites_visit(db_file, conn)
	except Exception as e:
		tqdm.write(f"Problem reading the sites_visits or the scraper data: {e}")
		exit()
	for i, row in tqdm(
			sites_visits.iterrows(),
			total=len(sites_visits),
			position=0,
			leave=True,
			ascii=True,
	):
		visit_id = row["visit_id"]
		site_url = row["site_url"]
		tqdm.write(f"• Visit ID: {visit_id} | Site URL: {site_url}")
		try:

			df_requests, df_responses, df_redirects, call_stacks, javascript = read_sql_crawl_data(visit_id, db_file, conn)

			# get the global_curr_domain
			global global_curr_domain;
			curr_domain_1 = get_fld(site_url)
			global_curr_domain.append(curr_domain_1)
			last_top_level_url = df_requests['top_level_url'].iloc[-1]
			curr_domain_2 = get_fld(last_top_level_url)
			global_curr_domain.append(curr_domain_2)
			# print("global_curr_domain: ", global_curr_domain)

			# Process and check each DataFrame (requests, responses, redirects)
			for df, url_column in [(df_requests, 'url'), (df_responses, 'url'), (df_redirects, 'old_request_url'),
								   (df_redirects, 'new_request_url')]:
				if df.empty:
					continue  # Skip processing if the DataFrame is empty
				df['temp_url'] = df[url_column]  # Temporary URL column

				df['is_ad'] = df['temp_url'].apply(label_data)  # Apply the ad checking function to each URL
				ad_urls = df[df['is_ad']]['temp_url']
				for url in ad_urls:
					print(url)
		except Exception as e:
			tqdm.write(f"Error: {e}")
			traceback.print_exc()
			pass


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Detect Ads"
	)
	parser.add_argument("--folder", type=str, default="data", help="the dataset folder")
	parser.add_argument("--output", type=str, default="output", help="the output folder")
	args = parser.parse_args()
	FOLDER = args.folder
	OUTPUT = args.output
	DB_FILE = os.path.join(FOLDER, f"datadir-0/crawl-data.sqlite")
	LDB_FILE = os.path.join(FOLDER, f"datadir-0/content.ldb")
	print("DB_FILE: ", DB_FILE)
	detect_ads(DB_FILE, LDB_FILE)


