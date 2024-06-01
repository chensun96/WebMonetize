from urllib.parse import urlparse
import pandas as pd
import re
import os


def affiliate_rule_1(url):
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
            (urlDomain == 'dotomi' and urlSubDomain == "cj") or
            (urlDomain == 'qksrv') or

            (urlDomain == 'zaful' and 'lkid' in params_list) or
            (urlDomain == 'codecanyon' and 'ref' in params_list) or
            (urlDomain == 'graphicriver' and 'ref' in params_list) or
            (urlDomain == 'themeforest' and 'ref' in params_list) or
            (urlDomain == 'admitad' and (regexp_admitad_1.search(urlPath) or regexp_admitad_2.search(x.urlPath))) or
            (urlDomain == 'flipkart' and 'affid' in params_list) or

            ((urlDomain == 'pntra' or
              urlDomain == 'gopjn' or
              urlDomain == 'pjtra' or
              urlDomain == 'pjatr' or
              urlDomain == 'pntrs' or
              urlDomain == 'pntrac') and (regexp_pepperjam.search(urlPath)))
    ):
      
        return True, urlDomain

    return False, ''


def affiliate_rule_2(url):
   
    pattern = r"(target.com/.*\?afid=)|(target\.com\/.*\bAFID=)|(target.com/.*&afid=)|(ad.admitad.com/g/)|(ad.admitad.com/goto/)|(performance.affiliaxe.com/.*\?aff_id=)|(performance.affiliaxe.com/.*&aff_id=)|(s.aliexpress.com/.*\?af=)|(s.aliexpress.com/.*&af=)|(amazon.com/.*\?tag=)|(amazon.com/.*&tag=)|(amazon.de/.*\?tag=)|(amazon.de/.*&tag=)|(amazon.it/.*\?tag=)|(amazon.it/.*&tag=)|(amazon.in/.*\?tag=)|(amazon.in/.*&tag=)|(amazon.fr/.*\?tag=)|(amazon.fr/.*&tag=)|(amazon\..*&tag=)|(amazon\..*?\?tag=)| (primevideo.com/.*\?ref=)|(primevideo.com/.*&ref=)|(itunes.apple.com/.*\?at=)|(itunes.apple.com/.*&at=)|(apple.com/.*\?afid=)|(apple.com/.*&afid=)|(affiliates.audiobooks.com/.*\?a_aid=.*&a_bid=)|(affiliates.audiobooks.com/.*\?a_bid=.*&a_aid=)|(affiliates.audiobooks.com/.*&a_bid=.*&a_aid=)|(avantlink.com/.*\?pw=)|(avantlink.com/.*&pw=)|(secure.avangate.com/.*\?affiliate=)|(secure.avangate.com/.*&affiliate=)|(awin1.com/.*\?awinaffid=)|(awin1.com/.*&awinaffid=)|(ad.zanox.com/ppc\^)|(zenaps.com/rclick.php\?)|(banggood.com/.*\?p=)|(banggood.com/.*&p=)|(bookdepository.com/.*\?a_aid=)|(bookdepository.com/.*&a_aid=)|(booking.com/.*\?aid=)|(booking.com/.*&aid=)|(hop.clickbank.net\^)|(anrdoezrs.net/click-)|(cj.dotomi.com\^)|(dpbolvw.net/click-)|(emjcd.com\^)|(jdoqocy.com/click-)|(kqzyfj.com/click-)|(qksrv.net\^)|(tkqlhce.com/click-)|(designmodo.com/\?\\u=)|(rover.ebay.com/.*\?campid=)|(rover.ebay.com/.*&campid=)|(audiojungle.net/.*\?ref=)|(audiojungle.net/.*&ref=)|(codecanyon.net/.*\?ref=)|(codecanyon.net/.*&ref=)|(marketplace.envato.com/.*\?ref=)|(marketplace.envato.com/.*&ref=)|(graphicriver.net/.*\?ref=)|(graphicriver.net/.*&ref=)|(themeforest.net/.*\?ref=)|(themeforest.net/.*&ref=)|(videohive.net/.*\?ref=)|(videohive.net/.*&ref=)|(buyeasy.by/cashback/)|(buyeasy.by/redirect/)|(flipkart.com/.*\?affid=)|(flipkart.com/.*&affid=)|(gtomegaracing.com/.*\?tracking=)|(gtomegaracing.com/.*&tracking=)|(search.hotellook.com/.*\?marker=)|(search.hotellook.com/.*&marker=)|(hotmart.net.br/.*\?a=)|(hotmart.net.br/.*&a=)|(7eer.net/c/)|(evyy.net/c/)|(kontrolfreek.com/.*\?a_aid=)|(kontrolfreek.com/.*&a_aid=)|(online.ladbrokes.com/promoRedirect\?\?key=)|(online.ladbrokes.com/promoRedirect\?.*&key=)|(makeupgeek.com/.*\?acc=)|(makeupgeek.com/.*&acc=)|(gopjn.com/t/)|(pjatr.com/t/)|(pjtra.com/t/)|(pntra.com/t/)|(pntrac.com/t/)|(pntrs.com/t/)|(click.linksynergy.com/.*\?id=)|(click.linksynergy.com/.*&id=)|(go.redirectingat.com/.*\?id=)|(go.redirectingat.com/.*&id=)|(olymptrade.com/.*\?affiliate)|(go\.skimresources\.com/\?id=\d+)|(c\.pepperjamnetwork\.com/click\?action=.*&sid=)|(avantlink\.com.*[?&;]tt=)|(events\.release\.narrativ\.com\/api\/v0\/)|(ojrq\.net\/p\/\?.*)"
    match = re.search(pattern, url)

    if match:
        print(f"Matched pattern: {match.group()} at position: {match.start()}-{match.end()}")
        return True, match.group()
    else:
        return False, ''

def affiliate_rule_3(url):
    pattern = r"(=aff$)|(\baff_id=)|(\baff_code=)|(\baff=\b)|(\bafftrack=)|(\baff_trace_key=)|(\baff_fcid=)|(\baff_fsk=)|(\baffgroup=)|(\baffiliatesgateway)|(utm_medium%3Daffiliate)|(\bmv=affiliate)|(\bas_channel=affiliate)|(\bPAffiliateID=)|(/discount/)|(\bcoupon=)"
    
    # Search for the pattern in the URL
    match = re.search(pattern, url)
    
    if match:
        print(f"Matched pattern: {match.group()} at position: {match.start()}-{match.end()}")
        return True, match.group()
    else:
        return False, ''


def affiliate_rule_4(url):
    pattern = r"(walmart\.com\/.*[?&]affiliates_ad_id=)|(amazon\.com\/.*\&tag=)|(amazon\.[a-z.]+\/.*[?&;]tag=[^&;]+)|(bestbuy\.com\/.*\?[^?]*irclickid=[^?]*&.*ref=)|(lego\.com\/.*AffiliateUS.*)|(mudpuppy\.com\/.*\?[^?]*irclickid=[^?]*&.*sharedid=)|(wayfair\.com\/.*\?[^?]*refID=[^?]*&.*clickid=)|(wayfair\.com\/.*\?[^?]*refid=)|(rei\.com\/.*\?[^?]*cm_mmc=aff_AL-.*)|(etsy\.com\/.*\?[^?]*sv_affiliateId=)|(homedepot\.com\/.*\?[^?]*cm_mmc=afl-ir-.*)|(lowes\.com\/.*\?[^?]*cm_mmc=aff-.*)|(potterybarn\.com\/.*\?[^?]*cm_ven=afshoppromo.*)|(potterybarnkids\.com\/.*\?[^?]*cm_ven=afshoppromo.*)|(utm_medium=affiliate[^&]*)|(utm_medium=Affiliate[^&]*)|(\butm_medium=Affiliate)|(utm_medium=affiliates)|(\bUtm_medium=affiliate[^&]*)|(\butm_source=affiliate[^&]*)|(\bsource=affiliate[^&]*)|(\butm_source=affiliates)|(\butm_medium=AFF[^&]*)|(nerdwallet\.com\/redirect\/[0-9a-fA-F-]+[?].*&impression_id=)|(\baffid=)|(\bAFFID=)|(\baffname=)|(\bAffsrc=)|(=IRAFF_[^&]*)|(\butm_medium=af)|(samsung\.com\/.*[?&]btn_ref=[^&]*)|(rstyle\.me/\+[^/])|(bhpho.to/.*)|(bhphotovideo.*\/BI\/)|(joinhoney\.com\/[a-zA-Z]+$)|(play.google.com/store/apps/details\?id=[^&]+&referrer=adjust_reftag[^&]*utm_source)|(play.google.com/store/apps/details\?id=[^&]+&referrer=af_tranid)|(keeps\.com\/.*\?utm_campaign=[^&]+)|(rayconglobal\.com\/.*\&utm_campaign=[^&]+)|(microcenter.com\/.*\?utm_campaign=[^&]+)\
        |(therealreal\.com\/.*\&utm_campaign=[^&]+)|(e\.lga\.to\/[a-zA-Z0-9_\-]+$)|(deezer\.com\/.*\&utm_campaign=[^&]+)|(hellofresh\.com\/.*\&utm_campaign=[^&]+)|(coinbase\.com\/.*\&utm_campaign=[^&]+)|(dbrand\.com\/.*\?utm_source=[^&]+)"
    
    # Search for the pattern in the URL
    match = re.search(pattern, url)
    
    if match:
        print(f"Matched pattern: {match.group()} at position: {match.start()}-{match.end()}")
        return True, match.group()
    else:
        return False, ''
    

def get_affiliate_rules():
    return [affiliate_rule_2, affiliate_rule_3, affiliate_rule_4, affiliate_rule_1]

def affiliate_rules(url):
    rules = [affiliate_rule_2, affiliate_rule_3, affiliate_rule_4, affiliate_rule_1]
  
    for rule in rules:
        matched = rule(url)
        if matched:
            return True
    return False,



def process_crawl(df_redirect_chains):
    # Apply affiliate_rules to each URL and record the specific rule matched
    # record the match.group() in the new column
    def apply_rules(url):
        rules = get_affiliate_rules()
        for rule in rules:
            matched, description = rule(url)
            if matched:
                return True, description
        return False, ''
    
    # Apply rules to each URL and store results

    df_redirect_chains['match_result'] = df_redirect_chains['url'].apply(apply_rules)
    df_redirect_chains['match'] = df_redirect_chains['match_result'].apply(lambda x: x[0])
    df_redirect_chains['rule_description'] = df_redirect_chains['match_result'].apply(lambda x: x[1])

    # Drop the temporary column and determine labels based on matches
    df_redirect_chains.drop(columns=['match_result'], inplace=True)
    
    # Map True/False to 'affiliate'/'others'
    df_redirect_chains['final_rules_based_label'] = df_redirect_chains.groupby('visit_id')['match'].transform(any).map({True: 'affiliate', False: 'others'})

    output_columns = ['visit_id', 'url', 'redirect_domain_total', 'match', 'rule_description', 'final_rules_based_label']
    df_rule_based_label = df_redirect_chains[output_columns]

    return df_rule_based_label
    

def apply_high_precision_rules(folder):
    for crawl_id in os.listdir(folder):
        each_crawl =  os.path.join(folder, crawl_id)
        
      

        # Check if this crawl complete, or still building
        #label_path = os.path.join(each_crawl, 'label.csv')
        #if not os.path.exists(label_path):
        #    print("This crawl is not complete. Ignore")
        #    continue

        rule_based_label_path = os.path.join(each_crawl, 'rule_based_label.csv')
        if not os.path.exists(rule_based_label_path):
            print("This crawl is been processing. Continue")
            continue

        redirect_chain_path = os.path.join(each_crawl, 'redirect_chains.csv')
        print("\nProcessing : ",redirect_chain_path)
        df_redirect_chains = pd.read_csv(redirect_chain_path, on_bad_lines='skip')
        df_rule_based_label  = process_crawl(df_redirect_chains)   
        df_rule_based_label.to_csv(rule_based_label_path, index=False)


def seperate_data_based_on_rule_labels(folder, rule_based_aff_folder, rule_based_others_folder):
    os.makedirs(folder, exist_ok=True)
    other_files = ['features_phase1_simple.csv', 'features_phase1.csv', 'records.csv', 'redirect_chains.csv', 'url_features.csv', 'rule_based_label.csv']
    data_files = {}
    
    for crawl_id in os.listdir(folder):
        each_crawl = os.path.join(folder, crawl_id)
        
        
        rule_based_label_path = os.path.join(each_crawl, 'rule_based_label.csv')
        if not os.path.exists(rule_based_label_path):
            print(f"This crawl {crawl_id} is not complete or rule_based_label.csv is missing. Ignoring.")
            continue

        df_labels = pd.read_csv(rule_based_label_path)
        if 'final_rules_based_label' not in df_labels.columns:
            print(f"Missing 'final_rules_based_label' in the data for crawl {crawl_id}. Ignoring.")
            continue

        # Pre-load other files into memory
        for file_name in other_files:
            file_path = os.path.join(each_crawl, file_name)
            if os.path.exists(file_path):
                data_files[file_name] = pd.read_csv(file_path)
        
        grouped = df_labels.groupby('visit_id')
        for visit_id, group in grouped:
            label = group['final_rules_based_label'].iloc[0]  # all rows in group have the same label
            
            if label == 'affiliate':
                destination_folder = os.path.join(rule_based_aff_folder, crawl_id)
            elif label == 'others':
                destination_folder = os.path.join(rule_based_others_folder, crawl_id)
            else:
                continue
            
            os.makedirs(destination_folder, exist_ok=True)
            
            # Handle files for this visit_id
            for file_name, df in data_files.items():
                df_filtered = df[df['visit_id'] == visit_id]
                # Check if any data exists for this visit_id
                if not df_filtered.empty:
                    other_destination_path = os.path.join(destination_folder, file_name)
                    if os.path.exists(other_destination_path):
                        df_filtered.to_csv(other_destination_path, mode='a', header=False, index=False)
                    else:
                        df_filtered.to_csv(other_destination_path, index=False)


    

if __name__ == "__main__":

    # step 1, apply high_precision_rules to all data
    #affiliate_folder = "../../output/affiliate_yt"
    #others_folder = "../../output/others_yt"
    #apply_high_precision_rules(affiliate_folder)
    #apply_high_precision_rules(others_folder)
    
    # step 2. Seperate data into two folders: "rule_based_aff", "rule_based_others" folder 
    #rule_based_aff_folder = "../../output/rule_based_aff_yt"
    #rule_based_others_folder = "../../output/rule_based_others_yt"
    #seperate_data_based_on_rule_labels(affiliate_folder, rule_based_aff_folder, rule_based_others_folder)
    #seperate_data_based_on_rule_labels(others_folder, rule_based_aff_folder, rule_based_others_folder)

    
    
    #all_folder = '/home/data/chensun/affi_project/purl/output/all_yt'
    #apply_high_precision_rules(all_folder)
    #rule_based_aff_folder = "../../output/rule_based_aff_yt"
    #rule_based_others_folder = "../../output/rule_based_others_yt"
    #seperate_data_based_on_rule_labels(all_folder, rule_based_aff_folder, rule_based_others_folder)
    
    
    
    
   
    # step 3. seperate other data






    url = 'https://play.google.com/store/books/details?id=ISBN_{book ISBN}&PAffiliateID={Your Affiliate ID}'
    
    matched, match_group = affiliate_rule_3(url)
    if not matched:
        print("No affiliate pattern matched.")
    else:
        print(match_group)
    

   
    #pattern = r"(pinterest\.com\/[a-zA-Z0-9_]+\/$)"
    #match = re.search(pattern, url)
    #if match:
    #    print(f"Matched pattern: {match.group()} at position: {match.start()}-{match.end()}")
        
    #else:
    #    print("not match")


