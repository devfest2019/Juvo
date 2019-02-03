import json
import requests

customerId = '5c561460322fa06b677944c8'
apiKey = 'dd87ae4d11280174539e4a48f46a8717'

def make_account(customer):
    url = 'http://api.reimaginebanking.com/customers/{}/accounts?key={}'.format(customerId,apiKey)
    payload = {
        "type": "Checking",
        "nickname": customer,
        "rewards": 0,
        "balance": 0,
    }

    # Create a Savings Account
    response = requests.post(
        url,
        data=json.dumps(payload),
        headers={'content-type':'application/json'}
    )

    if response.status_code == 201:
        print("Account creation successful")
        return url


def make_loan(url, ccs, cramount=100, tlength=12, cdescript="awesome loans"):
    payload = {
        "type": "small business",
        "status": "approved",
        "credit_score": ccs,
        "monthly_payment": cramount/tlength,
        "amount": cramount,
        "description": cdescript
    }

    # Create a Loan in Account
    response = requests.post(
        url,
        data=json.dumps(payload),
        headers={'content-type':'application/json'},
    )

    if response.status_code == 201:
        print('Loan created')

    return url;

    
# Use notebook to clean
def clean_up(url):
    pass
