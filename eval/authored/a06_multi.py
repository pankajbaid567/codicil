M = "multi"
ITEMS = [
    (
        "m01",
        "What is the no-AFA limit for recurring payments, and how many factors does a digital payment generally need?",
        [
            ("emandate", "All recurring transactions may be authorised without AFA up to ₹15,000/- per transaction"),
            ("auth", "All digital payment transactions shall be authenticated by at least two distinct factors of authentication"),
        ],
        None,
        M,
    ),
    (
        "m02",
        "How do bank and NBFC rules differ on recovery agents' conduct?",
        [
            ("rbc", "the bank shall not resort to undue harassment, viz., persistently bothering the borrower(s) at odd hours"),
            (
                "nbfcrbc",
                "properly trained to handle their responsibilities with care and sensitivity, particularly aspects such as soliciting customers, hours of calling",
            ),
        ],
        None,
        M,
    ),
    (
        "m03",
        "What must a card issuer do when a cardholder disputes a bill, and how are refunds credited?",
        [
            (
                "cc",
                "In case, a cardholder protests any bill, the card-issuer shall provide explanation and, wherever applicable, documentary evidence shall be provided to the cardholder within a maximum period of 30 days",
            ),
            ("cc", "shall be immediately adjusted against the ‘payment due’ and notified to the cardholder"),
        ],
        None,
        M,
    ),
    (
        "m04",
        "What are the limits on peer-to-peer loan maturity and on a lender's total exposure?",
        [
            ("p2p", "The maturity of the loans shall not exceed 36 months"),
            (
                "p2p",
                "The aggregate exposure of a lender to all borrowers at any point of time, across all NBFC-P2P platforms, shall be subject to a cap of ₹50,00,000/-",
            ),
        ],
        None,
        M,
    ),
    (
        "m05",
        "When is a customer's liability zero, and how quickly must the bank reverse an unauthorised transaction?",
        [
            ("rbc", "A customer’s entitlement to zero liability shall arise where the unauthorised transaction occurs"),
            (
                "rbc",
                "the bank shall credit (shadow reversal) the amount involved in the unauthorised electronic transaction to the customer’s account within 10 working days",
            ),
        ],
        None,
        M,
    ),
    (
        "m06",
        "What are the rules for video KYC and for periodic KYC updation?",
        [
            ("kyc", "The bank shall vary the sequence and / or type of questions, including those indicating the liveness of the interaction"),
            ("kyc", "The bank shall adopt a risk-based approach for periodic updation of KYC"),
        ],
        None,
        M,
    ),
    (
        "m07",
        "What must a prepaid wallet issuer do before the wallet expires and with the balance afterwards?",
        [
            ("ppi", "PPI issuer shall caution the PPI holder at reasonable intervals, during the 45 days’ period prior to expiry"),
            (
                "ppi",
                "Non-bank PPI issuer cannot transfer the outstanding balance to its Profit & Loss account for at least three years from the expiry date of PPI",
            ),
        ],
        None,
        M,
    ),
    (
        "m08",
        "What are the consent rules for unsolicited credit offers and for sharing data with credit bureaus?",
        [
            ("cc", "Unsolicited loans or other credit facilities shall not be offered to the credit cardholders without seeking explicit consent"),
            ("cc", "the card-issuer shall explicitly bring to the notice of the customer that such information is being provided"),
        ],
        None,
        M,
    ),
]
