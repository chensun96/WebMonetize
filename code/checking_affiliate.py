from urllib.parse import urlparse
import pandas as pd
import re
import graph_scripts as gs
import os

def apply_affiliate_rule(url):
    if apply_affiliate_rule_1(url) or apply_affiliate_rule_2(url) or apply_affiliate_rule_3(url):
        return True
    return False

def apply_affiliate_rule_1(url):
    # Parse the URL string using urlparse
    parsed_url = urlparse(url)
    # Extract the network location (netloc) and split it into parts
    netloc_parts = parsed_url.netloc.split('.')
    if len(netloc_parts) >= 3:
        # Typical case: subdomain.domain.tld
        urlSubDomain = '.'.join(netloc_parts[:-2]).lower()
        urlDomain = netloc_parts[-2].lower()
    elif len(netloc_parts) == 2:
        # No subdomain, only domain.tld
        urlSubDomain = ''
        urlDomain = netloc_parts[0].lower()
    else:
        # Handle edge cases (like 'localhost' or custom network locations)
        urlSubDomain = ''
        urlDomain = parsed_url.netloc.lower()
    if urlSubDomain.startswith('www.'):
        urlSubDomain = urlSubDomain[4:]
    
    urlPath = parsed_url.path.lower()
    urlParams = parsed_url.query.lower()
    #params_list = map(lambda y: y.split('=')[0], x.urlParams.split('&'))
    params_list = list(map(lambda y: y.split('=')[0], urlParams.split('&')))
    
    #print(f"Domain: {urlDomain}, Subdomain: {urlSubDomain}, Path: {urlPath}, Params: {urlParams}")
    #print(f"Params list: {params_list}")

    regexp_clickbank = re.compile(r'.*.hop')
    regexp_anrdoezrs = re.compile(r'/click-[0-9]+-[0-9]+')
    regexp_buyeasy_1 = re.compile(r'/cashback.*')
    regexp_buyeasy_2 = re.compile(r'/redirect.*')
    regexp_admitad_1 = re.compile(r'/g/.*')
    regexp_admitad_2 = re.compile(r'/goto/.*')
    regexp_impactradius = re.compile(r'/c/[0-9]+/[0-9]+/[0-9]+')
    regexp_pepperjam = re.compile(r'/t/[0-9]-[0-9]+-[0-9]+-[0-9]+')

    if ((urlDomain == 'affiliaxe' and 'aff_id' in params_list) or
            (urlDomain == 'aliexpress' and 'af' in params_list) or
            (urlDomain == 'amazon' and 'tag' in params_list) or
            (urlDomain == 'apessay' and 'rid' in params_list) or
            (urlDomain == 'apple' and 'at' in params_list) or
            (urlDomain == 'audiojungle' and 'ref' in params_list) or
            (urlDomain == 'awin1' and 'awinaffid' in params_list) or
            (urlDomain == 'zanox' and urlPath == '/ppc') or
            (urlDomain == 'zenaps' and urlPath == '/rclick.php') or
            (urlDomain == 'banggood' and 'p' in params_list) or
            (urlDomain == 'bookdepository' and 'a_aid' in params_list) or
            (urlDomain == 'booking' and 'aid' in params_list) or
            (urlDomain == 'ebay' and 'campid' in params_list) or
            (urlDomain == 'envato' and 'ref' in params_list) or
            (urlDomain == 'gtomegaracing' and 'tracking' in params_list) or
            (urlDomain == 'hotellook' and 'marker' in params_list) or
            (urlDomain == 'hotmart' and 'a' in params_list) or
            (urlDomain == 'kontrolfreek' and 'a_aid' in params_list) or
            (urlDomain == 'shareasale' and urlPath == '/r.cfm') or
            (urlDomain == 'shareasale' and urlPath == '/m-pr.cfm') or
            (urlDomain == 'shareasale' and urlPath == '/u.cfm') or
            (urlDomain == 'rstyle') or
            (urlDomain == 'shopstyle') or
            (urlDomain == 'makeupgeek' and 'acc' in params_list) or
            (urlDomain == 'olymptrade' and 'affiliate_id' in params_list) or
            (urlDomain == 'videohive' and 'ref' in params_list) or
            (urlDomain == 'avantlink' and 'pw' in params_list) or
            (urlDomain == 'avangate' and 'AFFILIATE'.lower() in params_list) or
            (urlDomain == 'redirectingat' and 'id' in params_list) or
            (urlDomain == 'linksynergy' and urlSubDomain == 'click' and 'id' in params_list) or
            (urlDomain == 'audiobooks' and 'a_aid' in params_list and 'a_bid' in params_list) or
            (urlDomain == 'buyeasy' and regexp_buyeasy_1.search(urlPath)) or
            (urlDomain == 'buyeasy' and regexp_buyeasy_2.search(urlPath)) or
            (urlDomain == 'clickbank' and regexp_clickbank.search(urlSubDomain)) or
            (urlDomain == 'aliexpress' and urlSubDomain == 's.click') or

            ((urlDomain == '7eer' or urlDomain == 'evyy' or urlDomain == 'ojrq') and regexp_impactradius.search(
                urlPath)) or

            ((
                     urlDomain == 'anrdoezrs' or urlDomain == 'dpbolvw' or urlDomain == 'kqzyfj' or urlDomain == 'jdoqocy' or urlDomain == 'tkqlhce') and regexp_anrdoezrs.search(
                urlPath)) or
            (urlDomain == 'emjcd') or
            (urlDomain == 'dotomi') or
            (urlDomain == 'qksrv') or

            (urlDomain == 'zaful' and 'lkid' in params_list) or
            (urlDomain == 'codecanyon' and 'ref' in params_list) or
            (urlDomain == 'graphicriver' and 'ref' in params_list) or
            (urlDomain == 'themeforest' and 'ref' in params_list) or
            (urlDomain == 'admitad' and (regexp_admitad_1.search(urlPath) or regexp_admitad_2.search(urlPath))) or
            (urlDomain == 'flipkart' and 'affid' in params_list) or

            ((urlDomain == 'pntra' or
              urlDomain == 'gopjn' or
              urlDomain == 'pjtra' or
              urlDomain == 'pjatr' or
              urlDomain == 'pntrs' or
              urlDomain == 'pntrac') and (regexp_pepperjam.search(urlPath)))
    ):
        return True

    return False

def apply_affiliate_rule_2(url):
    pattern = r"(target.com/.*\?afid=)|(target.com/.*&afid=)|(ad.admitad.com/g/)|(ad.admitad.com/goto/)|(performance.affiliaxe.com/.*\?aff_id=)|(performance.affiliaxe.com/.*&aff_id=)|(s.aliexpress.com/.*\?af=)|(s.aliexpress.com/.*&af=)|(amazon.com/.*\?tag=)|(amazon.com/.*&tag=)|(amazon.de/.*\?tag=)|(amazon.it/.*\?tag=)|(amazon.it/.*&tag=)|(amazon.in/.*\?tag=)|(amazon.in/.*&tag=)|(amazon.fr/.*\?tag=)|(amazon.fr/.*&tag=)|(primevideo.com/.*\?ref=)|(primevideo.com/.*&ref=)|(itunes.apple.com/.*\?at=)|(itunes.apple.com/.*&at=)|(apple.com/.*\?afid=)|(apple.com/.*&afid=)|(affiliates.audiobooks.com/.*\?a_aid=.*&a_bid=)|(affiliates.audiobooks.com/.*\?a_bid=.*&a_aid=)|(affiliates.audiobooks.com/.*&a_bid=.*&a_aid=)|(avantlink.com/.*\?pw=)|(avantlink.com/.*&pw=)|(secure.avangate.com/.*\?affiliate=)|(secure.avangate.com/.*&affiliate=)|(awin1.com/.*\?awinaffid=)|(awin1.com/.*&awinaffid=)|(ad.zanox.com/ppc\^)|(zenaps.com/rclick.php\?)|(banggood.com/.*\?p=)|(banggood.com/.*&p=)|(bookdepository.com/.*\?a_aid=)|(bookdepository.com/.*&a_aid=)|(booking.com/.*\?aid=)|(booking.com/.*&aid=)|(hop.clickbank.net\^)|(anrdoezrs.net/click-)|(cj.dotomi.com\^)|(dpbolvw.net/click-)|(emjcd.com\^)|(jdoqocy.com/click-)|(kqzyfj.com/click-)|(qksrv.net\^)|(tkqlhce.com/click-)|(designmodo.com/\?\\u=)|(rover.ebay.com/.*\?campid=)|(rover.ebay.com/.*&campid=)|(audiojungle.net/.*\?ref=)|(audiojungle.net/.*&ref=)|(codecanyon.net/.*\?ref=)|(codecanyon.net/.*&ref=)|(marketplace.envato.com/.*\?ref=)|(marketplace.envato.com/.*&ref=)|(graphicriver.net/.*\?ref=)|(graphicriver.net/.*&ref=)|(themeforest.net/.*\?ref=)|(themeforest.net/.*&ref=)|(videohive.net/.*\?ref=)|(videohive.net/.*&ref=)|(buyeasy.by/cashback/)|(buyeasy.by/redirect/)|(flipkart.com/.*\?affid=)|(flipkart.com/.*&affid=)|(gtomegaracing.com/.*\?tracking=)|(gtomegaracing.com/.*&tracking=)|(search.hotellook.com/.*\?marker=)|(search.hotellook.com/.*&marker=)|(hotmart.net.br/.*\?a=)|(hotmart.net.br/.*&a=)|(7eer.net/c/)|(evyy.net/c/)|(kontrolfreek.com/.*\?a_aid=)|(kontrolfreek.com/.*&a_aid=)|(online.ladbrokes.com/promoRedirect\?\?key=)|(online.ladbrokes.com/promoRedirect\?.*&key=)|(makeupgeek.com/.*\?acc=)|(makeupgeek.com/.*&acc=)|(gopjn.com/t/)|(pjatr.com/t/)|(pjtra.com/t/)|(pntra.com/t/)|(pntrac.com/t/)|(pntrs.com/t/)|(click.linksynergy.com/.*\?id=)|(click.linksynergy.com/.*&id=)|(go.redirectingat.com/.*\?id=)|(go.redirectingat.com/.*&id=)|(olymptrade.com/.*\?affiliate)"
    search_pattern = re.compile(pattern)

    if search_pattern.search(url):
        return True
    else:
        return False

def apply_affiliate_rule_3(url):
    pattern = r"(=aff$)|(\baff_id=)|(\baff_code=)|(\baff=\b)|(\bafftrack=)|(\bref_id=)|(\bref_id=\b)|(\brfsn=)|(\brcode=)|(\breferral\b)|(\baff_trace_key=)|(\baff_fcid=)|(\baff_fsk=)|(\baffiliate\b)"

    # Search for the pattern in the URL
    match = re.search(pattern, url)
    if re.search(pattern, url):
        print(f"Matched pattern: {match.group()} at position: {match.start()}-{match.end()}")
        
        return True  # The URL matches the pattern
    else:
        return False  # The URL does not match the pattern


def affiliate_rule_1(x):
    params_list = map(lambda y: y.split('=')[0], x.urlParams.split('&'))
    regexp_clickbank = re.compile(r'.*.hop')
    regexp_anrdoezrs = re.compile(r'/click-[0-9]+-[0-9]+')
    regexp_buyeasy_1 = re.compile(r'/cashback.*')
    regexp_buyeasy_2 = re.compile(r'/redirect.*')
    regexp_admitad_1 = re.compile(r'/g/.*')
    regexp_admitad_2 = re.compile(r'/goto/.*')
    regexp_impactradius = re.compile(r'/c/[0-9]+/[0-9]+/[0-9]+')
    regexp_pepperjam = re.compile(r'/t/[0-9]-[0-9]+-[0-9]+-[0-9]+')

    if ((x.urlDomain == 'affiliaxe' and 'aff_id' in params_list) or
            (x.urlDomain == 'aliexpress' and 'af' in params_list) or
            (x.urlDomain == 'amazon' and 'tag' in params_list) or
            (x.urlDomain == 'apessay' and 'rid' in params_list) or
            (x.urlDomain == 'apple' and 'at' in params_list) or
            (x.urlDomain == 'audiojungle' and 'ref' in params_list) or
            (x.urlDomain == 'awin1' and 'awinaffid' in params_list) or
            (x.urlDomain == 'zanox' and x.urlPath == '/ppc') or
            (x.urlDomain == 'zenaps' and x.urlPath == '/rclick.php') or
            (x.urlDomain == 'banggood' and 'p' in params_list) or
            (x.urlDomain == 'bookdepository' and 'a_aid' in params_list) or
            (x.urlDomain == 'booking' and 'aid' in params_list) or
            (x.urlDomain == 'ebay' and 'campid' in params_list) or
            (x.urlDomain == 'envato' and 'ref' in params_list) or
            (x.urlDomain == 'gtomegaracing' and 'tracking' in params_list) or
            (x.urlDomain == 'hotellook' and 'marker' in params_list) or
            (x.urlDomain == 'hotmart' and 'a' in params_list) or
            (x.urlDomain == 'kontrolfreek' and 'a_aid' in params_list) or
            (x.urlDomain == 'shareasale' and x.urlPath == '/r.cfm') or
            (x.urlDomain == 'shareasale' and x.urlPath == '/m-pr.cfm') or
            (x.urlDomain == 'shareasale' and x.urlPath == '/u.cfm') or
            (x.urlDomain == 'rstyle') or
            (x.urlDomain == 'shopstyle') or
            (x.urlDomain == 'makeupgeek' and 'acc' in params_list) or
            (x.urlDomain == 'olymptrade' and 'affiliate_id' in params_list) or
            (x.urlDomain == 'videohive' and 'ref' in params_list) or
            (x.urlDomain == 'avantlink' and 'pw' in params_list) or
            (x.urlDomain == 'avangate' and 'AFFILIATE'.lower() in params_list) or
            (x.urlDomain == 'redirectingat' and 'id' in params_list) or
            (x.urlDomain == 'linksynergy' and x.urlSubDomain == 'click' and 'id' in params_list) or
            (x.urlDomain == 'audiobooks' and 'a_aid' in params_list and 'a_bid' in params_list) or
            (x.urlDomain == 'buyeasy' and regexp_buyeasy_1.search(x.urlPath)) or
            (x.urlDomain == 'buyeasy' and regexp_buyeasy_2.search(x.urlPath)) or
            (x.urlDomain == 'clickbank' and regexp_clickbank.search(x.urlSubDomain)) or
            (x.urlDomain == 'aliexpress' and x.urlSubDomain == 's.click') or

            ((x.urlDomain == '7eer' or x.urlDomain == 'evyy' or x.urlDomain == 'ojrq') and regexp_impactradius.search(
                x.urlPath)) or

            ((
                     x.urlDomain == 'anrdoezrs' or x.urlDomain == 'dpbolvw' or x.urlDomain == 'kqzyfj' or x.urlDomain == 'jdoqocy' or x.urlDomain == 'tkqlhce') and regexp_anrdoezrs.search(
                x.urlPath)) or
            (x.urlDomain == 'emjcd') or
            (x.urlDomain == 'dotomi' and x.sulSubDomain == "cj") or
            (x.urlDomain == 'qksrv') or

            (x.urlDomain == 'zaful' and 'lkid' in params_list) or
            (x.urlDomain == 'codecanyon' and 'ref' in params_list) or
            (x.urlDomain == 'graphicriver' and 'ref' in params_list) or
            (x.urlDomain == 'themeforest' and 'ref' in params_list) or
            (x.urlDomain == 'admitad' and (regexp_admitad_1.search(x.urlPath) or regexp_admitad_2.search(x.urlPath))) or
            (x.urlDomain == 'flipkart' and 'affid' in params_list) or

            ((x.urlDomain == 'pntra' or
              x.urlDomain == 'gopjn' or
              x.urlDomain == 'pjtra' or
              x.urlDomain == 'pjatr' or
              x.urlDomain == 'pntrs' or
              x.urlDomain == 'pntrac') and (regexp_pepperjam.search(x.urlPath)))
    ):
        return True

    return False

# check from the url contains affiliate key words
def affiliate_rule_2(row, url_column):

    url = row.get(url_column, '').lower()  # Get the URL and convert it to lowercase
    # pattern = r"(=aff)|(aff_id=)|(aff_code=)|(aff=)|(=ref)|(afftrack=)|(ref_id=)|(ref_id=)| (rfsn=)|(rcode=)|(referral)|(aff_trace_key=)|(aff_fcid=)|(aff_fsk=)|(affiliate)"
    pattern1 = r"(=aff$)|(\baff_id=)|(\baff_code=)|(\baff=\b)|(\=ref\b)|(\bafftrack=)|(\bref_id=)|(\bref_id=\b)|(\brfsn=)|(\brcode=)|(\breferral\b)|(\baff_trace_key=)|(\baff_fcid=)|(\baff_fsk=)|(\baffiliate\b)"
    pattern2 = r"(target.com/.*\?afid=)|(target.com/.*&afid=)|(ad.admitad.com/g/)|(ad.admitad.com/goto/)|(performance.affiliaxe.com/.*\?aff_id=)|(performance.affiliaxe.com/.*&aff_id=)|(s.aliexpress.com/.*\?af=)|(s.aliexpress.com/.*&af=)|(amazon.com/.*\?tag=)|(amazon.com/.*&tag=)|(amazon.de/.*\?tag=)|(amazon.de/.*&tag=)|(amazon.it/.*\?tag=)|(amazon.it/.*&tag=)|(amazon.in/.*\?tag=)|(amazon.in/.*&tag=)|(amazon.fr/.*\?tag=)|(amazon.fr/.*&tag=)|(primevideo.com/.*\?ref=)|(primevideo.com/.*&ref=)|(itunes.apple.com/.*\?at=)|(itunes.apple.com/.*&at=)|(apple.com/.*\?afid=)|(apple.com/.*&afid=)|(affiliates.audiobooks.com/.*\?a_aid=.*&a_bid=)|(affiliates.audiobooks.com/.*\?a_bid=.*&a_aid=)|(affiliates.audiobooks.com/.*&a_bid=.*&a_aid=)|(avantlink.com/.*\?pw=)|(avantlink.com/.*&pw=)|(secure.avangate.com/.*\?affiliate=)|(secure.avangate.com/.*&affiliate=)|(awin1.com/.*\?awinaffid=)|(awin1.com/.*&awinaffid=)|(ad.zanox.com/ppc\^)|(zenaps.com/rclick.php\?)|(banggood.com/.*\?p=)|(banggood.com/.*&p=)|(bookdepository.com/.*\?a_aid=)|(bookdepository.com/.*&a_aid=)|(booking.com/.*\?aid=)|(booking.com/.*&aid=)|(hop.clickbank.net\^)|(anrdoezrs.net/click-)|(cj.dotomi.com\^)|(dpbolvw.net/click-)|(emjcd.com\^)|(jdoqocy.com/click-)|(kqzyfj.com/click-)|(qksrv.net\^)|(tkqlhce.com/click-)|(designmodo.com/\?\\u=)|(rover.ebay.com/.*\?campid=)|(rover.ebay.com/.*&campid=)|(audiojungle.net/.*\?ref=)|(audiojungle.net/.*&ref=)|(codecanyon.net/.*\?ref=)|(codecanyon.net/.*&ref=)|(marketplace.envato.com/.*\?ref=)|(marketplace.envato.com/.*&ref=)|(graphicriver.net/.*\?ref=)|(graphicriver.net/.*&ref=)|(themeforest.net/.*\?ref=)|(themeforest.net/.*&ref=)|(videohive.net/.*\?ref=)|(videohive.net/.*&ref=)|(buyeasy.by/cashback/)|(buyeasy.by/redirect/)|(flipkart.com/.*\?affid=)|(flipkart.com/.*&affid=)|(gtomegaracing.com/.*\?tracking=)|(gtomegaracing.com/.*&tracking=)|(search.hotellook.com/.*\?marker=)|(search.hotellook.com/.*&marker=)|(hotmart.net.br/.*\?a=)|(hotmart.net.br/.*&a=)|(7eer.net/c/)|(evyy.net/c/)|(kontrolfreek.com/.*\?a_aid=)|(kontrolfreek.com/.*&a_aid=)|(online.ladbrokes.com/promoRedirect\?\?key=)|(online.ladbrokes.com/promoRedirect\?.*&key=)|(makeupgeek.com/.*\?acc=)|(makeupgeek.com/.*&acc=)|(gopjn.com/t/)|(pjatr.com/t/)|(pjtra.com/t/)|(pntra.com/t/)|(pntrac.com/t/)|(pntrs.com/t/)|(click.linksynergy.com/.*\?id=)|(click.linksynergy.com/.*&id=)|(go.redirectingat.com/.*\?id=)|(go.redirectingat.com/.*&id=)|(olymptrade.com/.*\?affiliate)"
    pattern3 = r"(walmart\.com\/.*[?&]affiliates_ad_id=)|(amazon\.com\/.*\&tag=)|(bestbuy\.com\/.*\?[^?]*irclickid=[^?]*&.*ref=)|(lego\.com\/.*AffiliateUS.*)|(mudpuppy\.com\/.*\?[^?]*irclickid=[^?]*&.*sharedid=)|(wayfair\.com\/.*\?[^?]*refID=[^?]*&.*clickid=)|(wayfair\.com\/.*\?[^?]*refid=)|(rei\.com\/.*\?[^?]*cm_mmc=aff_AL-.*)|(etsy\.com\/.*\?[^?]*sv_affiliateId=)|(homedepot\.com\/.*\?[^?]*cm_mmc=afl-ir-.*)|(lowes\.com\/.*\?[^?]*cm_mmc=aff-.*)|(potterybarn\.com\/.*\?[^?]*cm_ven=afshoppromo.*)|(potterybarnkids\.com\/.*\?[^?]*cm_ven=afshoppromo.*)|(\butm_medium=affiliate[^&]*)|(\bUtm_medium=affiliate[^&]*)|(\butm_source=affiliate[^&]*)|(\bsource=affiliate[^&]*)|(\butm_medium=AFF[^&]*)|(nerdwallet\.com\/redirect\/[0-9a-fA-F-]+[?].*&impression_id=)|(\baffid=)|(\baffname=)|([?&].*=IRAFF_[^&]*)|(samsung\.com\/.*[?&]btn_ref=[^&]*)"
    # Search for the pattern in the URL
    match1 = re.search(pattern1, url)
    match2 = re.search(pattern2, url)
    match3 = re.search(pattern3, url)
    if re.search(pattern1, url):
        print(f"Matched pattern: {match1.group()} at position: {match1.start()}-{match1.end()}")
        return True  # The URL matches the pattern
    elif re.search(pattern2, url):
        print(f"Matched pattern: {match2.group()} at position: {match2.start()}-{match2.end()}")
        return True  # The URL matches the pattern
    elif re.search(pattern3, url):
        print(f"Matched pattern: {match3.group()} at position: {match3.start()}-{match3.end()}")
        return True  # The URL matches the pattern
    else:
        return False  # The URL does not match the pattern

def manipulate_urls(row, url_column):
    url = row.get(url_column, '')
    parsed_url = urlparse(url)
    # Extract the network location (netloc) and split it into parts
    netloc_parts = parsed_url.netloc.split('.')

    # Identify the subdomain part (assuming a common format of subdomain.domain.tld)
    # This will work for typical URLs, but might need adjustment for different URL formats
    # Extract the subdomain and domain
    if len(netloc_parts) >= 3:
        # Typical case: subdomain.domain.tld
        sd = '.'.join(netloc_parts[:-2]).lower()
        domain = netloc_parts[-2].lower()
    elif len(netloc_parts) == 2:
        # No subdomain, only domain.tld
        sd = ''
        domain = netloc_parts[0].lower()
    else:
        # Handle edge cases (like 'localhost' or custom network locations)
        sd = ''
        domain = parsed_url.netloc.lower()
    if sd.startswith('www.'):
        sd = sd[4:]
    # Return the required components
    return pd.Series([domain,
                      parsed_url.path.lower(),
                      sd,
                      parsed_url.query.lower()])


def check_affiliate_link(visit_id, site_url, conn):
    affiliate_link_found = False
    df_requests, df_responses, df_redirects, call_stacks, javascript = gs.read_tables_phase1(conn, visit_id)

    # Process and check each DataFrame (requests, responses, redirects)
    for df, url_column in [(df_requests, 'url'), (df_responses, 'url'), (df_redirects, 'old_request_url'), (df_redirects, 'new_request_url')]:
        # Skip processing if the DataFrame is empty
        if df.empty:
            continue
        df['temp_url'] = df[url_column]  # Temporary URL column

        # apply affiliate rule 1 and 2
        df[['urlDomain', 'urlPath', 'urlSubDomain', 'urlParams']] = df.apply(lambda x: manipulate_urls(x, 'temp_url'), axis=1)
        df['affiliateLink'] = df.apply(affiliate_rule_1, axis=1)
        df['affiliateLink_2'] = df.apply(affiliate_rule_2, url_column='temp_url', axis=1)
        df.drop('temp_url', axis=1, inplace=True)  # Remove the temporary column

        # Check if an affiliate link is found. Once found, no need to check other tables, return immediately.
        if df['affiliateLink'].any() or df['affiliateLink_2'].any():

            # Temporarily adjust the display settings for large width
            pd.set_option('display.max_colwidth', None)

            affiliate_link_found = True
            affiliate_links = df[(df['affiliateLink'] == True) | (df['affiliateLink_2'] == True)]
            affiliate_links['visit_id'] = visit_id
            affiliate_links['url'] = site_url
            affiliate_links['aff_url'] = df[url_column]
            affiliate_links = affiliate_links[['visit_id', 'url', 'aff_url']]
            all_affiliate_url = []
            for url in affiliate_links['aff_url']:
                all_affiliate_url.append(url)
                print("aff_url: ",url)

            # clean the combined_affiliate_links
            affiliate_links = affiliate_links.drop_duplicates(subset=['visit_id', 'url'], keep='first')

            print ("This is the intermediate affiliate url\n", affiliate_links['aff_url'])

            # save the affiliate or normal link into csv files
            affiliate_link = "/home/data/chensun/affi_project/purl/code/affiliate_potential.csv"
            columns = ["site_url", "aff_url"]

            df = pd.DataFrame([[site_url, all_affiliate_url]], columns=columns)
            if not os.path.exists(affiliate_link):
                df.to_csv(affiliate_link, index=False)
            else:
                df.to_csv(affiliate_link, mode='a', header=False, index=False)

            # Reset the option to its default after printing
            pd.reset_option('display.max_colwidth')


            return affiliate_link_found

    return affiliate_link_found



# Example usage
# url = "https://pubmatic-match.dotomi.com/match/bounce/current?networkId=17100&version=1&nuid=C95B37D1-F4A5-42DC-A3C4-39E42FF6F547&gdpr=0&gdpr_consent="
# rint(apply_affiliate_rule_1(url))

