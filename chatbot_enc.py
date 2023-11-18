from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load pre-trained embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
loaded_embedding_array = np.load("sentence_embeddings_minilm.npy")
sentences = [
{"question":" What is PrimeTrader?", "answer":" PrimeTrader is a decentralized trading platform that enables users to manage and trade digital assets while participating in trading competitions. It leverages blockchain technology and a community of traders to provide a unique trading experience."},

{"question":" How does PT differ from other trading platforms?", "answer":" PrimeTrader distinguishes itself by offering a competitive trading environment through trading competitions. It incorporates web3 technologies, including blockchain, NFTs, and staking for PTT tokens. The platform fosters a community of traders and emphasizes both individual trading and social engagement."},

{"question":" How do I create an account on PrimeTrader?", "answer":" To create an account, visit the PrimeTrader website, and follow the registration process, which typically involves providing your email, setting a password, and possibly connecting your web3 wallet."},

{"question":" Are there any fees associated with setting up an account or trading on PrimeTrader?", "answer":" PrimeTrader may charge nominal fees for certain transactions, but account creation is generally free. The specific fee structure can be found in the platform's terms and conditions."},

{"question":" Is PrimeTrader available globally?", "answer":" Yes, PrimeTrader is typically accessible to traders worldwide, subject to any legal or regulatory restrictions in specific regions."},

{"question":" Can I access PrimeTrader on both mobile and desktop?", "answer":" Yes, PrimeTrader is designed to be accessible on both mobile and desktop devices, ensuring flexibility for users."},

{"question":" Can I connect my Web3 wallet?", "answer":" Yes, you can connect your web3 wallet to PrimeTrader, allowing you to securely manage and trade digital assets directly from your wallet."},

{"question":" What is the PrimeTrader Token (PTT)?", "answer":" The PrimeTrader Token (PTT) is the native utility token of the platform. It can be used for various purposes within the ecosystem."},

{"question":" How is PTT used on the platform?", "answer":" PTT can be used for trading competitions, staking, and potentially as a means of payment for fees and services on PrimeTrader."},

{"question":" How can I earn PTT?", "answer":" You can earn PTT through trading competitions, staking, referrals, or other reward mechanisms specified by PrimeTrader."},

{"question":" Is PTT listed on any exchanges?", "answer":" Information about the listing of PTT on exchanges can be found on the PrimeTrader platform or the official website. PTT may be listed on both centralized and decentralized exchanges."},

{"question":" What is an investment in the context of PrimeTrader?", "answer":" In the context of PrimeTrader, investment often refers to staking PTT tokens to earn rewards, such as additional PTT or other benefits."},

{"question":" How do I stake my PTT tokens?", "answer":" Staking PTT tokens typically involves locking them in a smart contract on the platform. Detailed instructions for staking are usually provided within the PrimeTrader platform."},

{"question":" What are the benefits of staking PTT tokens on PrimeTrader?", "answer":" Staking PTT tokens can provide benefits such as earning staking rewards, participating in governance decisions, and potentially gaining access to exclusive features or competitions."},

{"question":" How are staking rewards calculated?", "answer":" Staking rewards are often calculated based on factors like the number of tokens staked, the duration of the stake, and the overall platform performance. The specific calculation details can be found on PrimeTrader."},

{"question":" Can I unstake my tokens at any time? Are there any penalties or lock-up periods?", "answer":" Unstaking terms, including lock-up periods and penalties, may vary. PrimeTrader should provide information on unstaking conditions, so users are aware of the terms before staking."},

{"question":" Are there any minimum or maximum limits to staking?", "answer":" Staking limits can vary based on the platform's policies and the specific staking pool in which you participate. PrimeTrader should provide guidance on minimum and maximum staking amounts."},

{"question":" What measures does PrimeTrader implement to ensure the safety of staked tokens?", "answer":" PrimeTrader typically employs advanced security measures and audits for smart contracts to ensure the safety of staked tokens. Specific security details may be outlined in the platform's documentation."},

{"question":" Can I stake tokens other than PTT on PrimeTrader?", "answer":" PrimeTrader may allow users to stake other compatible tokens, but the availability of such features depends on the platform's policies."},

{"question":" How does staking interact with the echo trading model?", "answer":" The interaction between staking and echo trading can vary. Staking may provide benefits or additional features within the echo trading system, but the specific details would be provided within the PrimeTrader platform."},

{"question":" How does PrimeTrader incorporate blockchain technology into its platform?", "answer":" PrimeTrader leverages blockchain technology to enable decentralized trading, transparent transactions, and secure management of digital assets. Smart contracts and other blockchain components play a crucial role."},

{"question":" What are NFTs, and how do they function on PrimeTrader?", "answer":" NFTs, or Non-Fungible Tokens, are unique digital assets. On PrimeTrader, NFTs may represent ownership of certain in-game assets, trading achievements, or other exclusive items. They can be traded, owned, and used for various purposes within the ecosystem."},

{"question":" How does echo trading work on PrimeTrader?", "answer":" Echo trading involves mirroring the trades of top traders on the platform. Users can choose to follow these traders, and their portfolios will replicate the selected trader's actions automatically."},

{"question":" Can you explain the on-chain order book system?", "answer":" The on-chain order book is a decentralized mechanism for matching buy and sell orders in a transparent and trustless manner. It ensures that trading activities are conducted securely and without the need for a centralized intermediary."},

{"question":" How does PrimeTrader utilize AI in its operations?", "answer":" PrimeTrader may use AI algorithms for various purposes, including data analysis, market insights, risk assessment, and automated trading strategies. The specific use of AI will depend on the platform's features and functionalities."},

{"question":" How secure are my information and assets on PrimeTrader?", "answer":" PrimeTrader prioritizes security and employs advanced encryption and security measures to protect user information and assets. Best practices are followed to ensure data and asset safety."},

{"question":" What measures does PrimeTrader take to ensure the authenticity of NFTs?", "answer":" NFT authenticity is typically guaranteed through blockchain technology, ensuring the uniqueness and provenance of each NFT. PrimeTrader likely uses blockchain verification to validate the authenticity of NFTs."},

{"question":" How is my trading history protected and tokenized on the platform?", "answer":" Trading history is often stored on the blockchain, providing transparency and security. Users can access their complete, immutable trading history through the blockchain."},

{"question":" What is Echo Trading?", "answer":" Echo Trading is a feature that allows users to automatically mirror the trading strategies and actions of experienced traders on the platform."},

{"question":" How can I start following top traders on PrimeTrader?", "answer":" To follow top traders, you can typically browse the list of available traders, review their performance, and choose to follow the ones that align with your trading preferences."},

{"question":" What determines a trader's Assets Under Management (AUM) on PrimeTrader?", "answer":" A trader's AUM is the total value of assets being managed or traded by that trader's followers. It includes the assets being mirrored by those following the trader."},

{"question":" How do the NFT levels impact AUM?", "answer":" NFT levels may grant certain privileges or benefits, which can potentially attract more followers. This, in turn, can increase a trader's AUM and potentially lead to higher earnings."},

{"question":" If I'm a top trader, how can I monetize my trading skills on PrimeTrader?", "answer":" Top traders on PrimeTrader can earn by attracting followers who choose to mirror their trading strategies. A percentage of the profits generated by followers may be distributed to the top trader."},

{"question":" What types of trading games can I participate in on PrimeTrader?", "answer":" PrimeTrader offers various trading competitions, which may include competitions against the market, peer-to-peer challenges, and other formats. The specific types of games can vary."},

{"question":" How does the Trader vs. Market game work?", "answer":" In the Trader vs. Market game, traders compete against the market's performance. Traders aim to outperform the market within a specified time frame to win rewards."},

{"question":" Can I challenge other traders directly?", "answer":" Yes, PrimeTrader typically allows traders to challenge each other directly in trading competitions. These challenges can be initiated within the platform."},

{"question":" How often are trading competitions held, and how can I participate?", "answer":" The frequency of trading competitions can vary, but PrimeTrader generally hosts them regularly. To participate, you can usually enter the competition through the platform, following the provided instructions."},

{"question":" What are the rewards for winning a trading competition?", "answer":" The rewards for winning a trading competition may include PTT tokens, NFTs, and other digital assets, depending on the competition's format and rules."},

{"question":" How can I mint, purchase, or trade NFTs on PrimeTrader?", "answer":" The process of minting, purchasing, or trading NFTs on PrimeTrader can typically be done through the platform's NFT marketplace. Detailed instructions are provided on the platform."},

{"question":" What benefits do different NFT levels offer?", "answer":" Different NFT levels may provide various benefits, such as exclusive access to trading features, reduced fees, or enhanced visibility on the platform. The specific benefits associated with each level will be outlined on PrimeTrader."},

{"question":" How does the staking and reward system work for PTT tokens?", "answer":" Staking PTT tokens may grant access to different NFT levels, which, in turn, can provide rewards, privileges, and potential earnings within the PrimeTrader ecosystem. Details can be found within the platform."},

{"question":" How can I connect with other traders or join the PrimeTrader community?", "answer":" You can typically engage with other traders and the PrimeTrader community through social features within the platform, such as chat rooms, forums, and social media channels."},

{"question":" Are there any educational resources available to help me become a better trader?", "answer":" PrimeTrader often provides educational resources, including tutorials, articles, and videos, to help users improve their trading skills and knowledge."},

{"question":" How can I participate in or host a trading tournament on PrimeTrader?", "answer":" You can participate in trading tournaments by following the instructions provided on the platform. If you're interested in hosting a tournament, you can likely contact the PrimeTrader team to discuss the process."},

{"question":" How can I become an affiliate or partner with PrimeTrader?", "answer":" Information on becoming an affiliate or partner with PrimeTrader can usually be found on the platform's website. Contact details or application forms may be available for interested parties."},

  {"question": "Could you tell me about PrimeTrader?", "answer": "Certainly! PrimeTrader is a decentralized trading platform designed for users to manage and trade digital assets. The platform incorporates blockchain technology and fosters a community of traders, offering a unique trading experience through competitions."},


  {"question": "What sets PrimeTrader apart from other trading platforms?", "answer": "PrimeTrader stands out by offering a competitive trading environment with trading competitions. It incorporates web3 technologies like blockchain, NFTs, and staking for PTT tokens. The platform emphasizes both individual trading and social engagement."},

  {"question": "How do I create an account on PrimeTrader?", "answer": "To create an account, visit the PrimeTrader website and follow the registration process. This typically involves providing your email, setting a password, and possibly connecting your web3 wallet."},

  {"question": "What's the process for setting up an account on PrimeTrader?", "answer": "Setting up an account on PrimeTrader involves visiting the platform's website and following the registration process. This usually includes providing your email, setting a password, and potentially connecting your web3 wallet."},

  {"question": "Are there any fees associated with setting up an account or trading on PrimeTrader?", "answer": "PrimeTrader may charge nominal fees for certain transactions, but account creation is generally free. The specific fee structure can be found in the platform's terms and conditions."},

  {"question": "Are there costs involved in creating an account or making trades on PrimeTrader?", "answer": "While PrimeTrader may charge nominal fees for specific transactions, creating an account is generally free. Detailed information about the fee structure can be found in the platform's terms and conditions."},

  {"question": "Is PrimeTrader available globally?", "answer": "Yes, PrimeTrader is typically accessible to traders worldwide, subject to any legal or regulatory restrictions in specific regions."},

  {"question": "Can users from around the world access PrimeTrader?", "answer": "Indeed, PrimeTrader is designed to be accessible globally, allowing users from around the world to trade. However, access may be subject to compliance with legal or regulatory restrictions in specific regions."},

  {"question": "Can I access PrimeTrader on both mobile and desktop?", "answer": "Yes, PrimeTrader is designed to be accessible on both mobile and desktop devices, providing flexibility for users."},

  {"question": "Is PrimeTrader compatible with both mobile and desktop devices?", "answer": "Certainly! PrimeTrader is designed to be compatible with both mobile and desktop devices, ensuring users can access the platform with flexibility."},

  {"question": "Can I connect my Web3 wallet?", "answer": "Yes, you can connect your Web3 wallet to PrimeTrader, allowing you to securely manage and trade digital assets directly from your wallet."},

  {"question": "Is it possible to link my Web3 wallet to PrimeTrader?", "answer": "Absolutely! You can link your Web3 wallet to PrimeTrader, providing a secure way to manage and trade digital assets directly from your wallet."},

  {"question": "What is the PrimeTrader Token (PTT)?", "answer": "The PrimeTrader Token (PTT) is the native utility token of the platform. It can be used for various purposes within the ecosystem."},

  {"question": "Could you explain the purpose of the PrimeTrader Token (PTT)?", "answer": "Certainly! The PrimeTrader Token (PTT) serves as the native utility token of the platform, offering various use cases within the PrimeTrader ecosystem."},

  {"question": "How is PTT used on the platform?", "answer": "PTT can be used for trading competitions, staking, and potentially as a means of payment for fees and services on PrimeTrader."},

  {"question": "In what ways can I use PTT on PrimeTrader?", "answer": "You can use PTT for trading competitions, staking, and potentially as a means of payment for fees and services on PrimeTrader."},

  {"question": "How can I earn PTT?", "answer": "You can earn PTT through trading competitions, staking, referrals, or other reward mechanisms specified by PrimeTrader."},

  {"question": "What are the avenues for earning PTT on PrimeTrader?", "answer": "You can earn PTT through various means, including trading competitions, staking, referrals, and other reward mechanisms specified by PrimeTrader."},

  {"question": "Is PTT listed on any exchanges?", "answer": "Information about the listing of PTT on exchanges can be found on the PrimeTrader platform or the official website. PTT may be listed on both centralized and decentralized exchanges."},

  {"question": "What is an investment in the context of PrimeTrader?", "answer": "In the context of PrimeTrader, investment often refers to staking PTT tokens to earn rewards, such as additional PTT or other benefits."},

  {"question": "How do I stake my PTT tokens?", "answer": "Staking PTT tokens typically involves locking them in a smart contract on the platform. Detailed instructions for staking are usually provided within the PrimeTrader platform."},

  {"question": "What are the benefits of staking PTT tokens on PrimeTrader?", "answer": "Staking PTT tokens can provide benefits such as earning staking rewards, participating in governance decisions, and potentially gaining access to exclusive features or competitions."},

  {"question": "How are staking rewards calculated?", "answer": "Staking rewards are often calculated based on factors like the number of tokens staked, the duration of the stake, and the overall platform performance. The specific calculation details can be found on PrimeTrader."},

  {"question": "Can I unstake my tokens at any time? Are there any penalties or lock-up periods?", "answer": "Unstaking terms, including lock-up periods and penalties, may vary. PrimeTrader should provide information on unstaking conditions, so users are aware of the terms before staking."},

  {"question": "Are there any minimum or maximum limits to staking?", "answer": "Staking limits can vary based on the platform's policies and the specific staking pool in which you participate. PrimeTrader should provide guidance on minimum and maximum staking amounts."},

  {"question": "What measures does PrimeTrader implement to ensure the safety of staked tokens?", "answer": "PrimeTrader typically employs advanced security measures and audits for smart contracts to ensure the safety of staked tokens. Specific security details may be outlined in the platform's documentation."},

  {"question": "Can I stake tokens other than PTT on PrimeTrader?", "answer": "PrimeTrader may allow users to stake other compatible tokens, but the availability of such features depends on the platform's policies."},

  {"question": "How does staking interact with the echo trading model?", "answer": "The interaction between staking and echo trading can vary. Staking may provide benefits or additional features within the echo trading system, but the specific details would be provided within the PrimeTrader platform."},

  {"question": "How does PrimeTrader incorporate blockchain technology into its platform?", "answer": "PrimeTrader leverages blockchain technology to enable decentralized trading, transparent transactions, and secure management of digital assets. Smart contracts and other blockchain components play a crucial role."},

  {"question": "What are NFTs, and how do they function on PrimeTrader?", "answer": "NFTs, or Non-Fungible Tokens, are unique digital assets. On PrimeTrader, NFTs may represent ownership of certain in-game assets, trading achievements, or other exclusive items. They can be traded, owned, and used for various purposes within the ecosystem."},

  {"question": "How does echo trading work on PrimeTrader?", "answer": "Echo trading involves mirroring the trades of top traders on the platform. Users can choose to follow these traders, and their portfolios will replicate the selected trader's actions automatically."},

  {"question": "Can you explain the on-chain order book system?", "answer": "The on-chain order book is a decentralized mechanism for matching buy and sell orders in a transparent and trustless manner. It ensures that trading activities are conducted securely and without the need for a centralized intermediary."},

  {"question": "How does PrimeTrader utilize AI in its operations?", "answer": "PrimeTrader may use AI algorithms for various purposes, including data analysis, market insights, risk assessment, and automated trading strategies. The specific use of AI will depend on the platform's features and functionalities."},

  {"question": "How secure are my information and assets on PrimeTrader?", "answer": "PrimeTrader prioritizes security and employs advanced encryption and security measures to protect user information and assets. Best practices are followed to ensure data and asset safety."},

  {"question": "What measures does PrimeTrader take to ensure the authenticity of NFTs?", "answer": "NFT authenticity is typically guaranteed through blockchain technology, ensuring the uniqueness and provenance of each NFT. PrimeTrader likely uses blockchain verification to validate the authenticity of NFTs."},

  {"question": "How is my trading history protected and tokenized on the platform?", "answer": "Trading history is often stored on the blockchain, providing transparency and security. Users can access their complete, immutable trading history through the blockchain."},

  {"question": "What is Echo Trading?", "answer": "Echo Trading is a feature that allows users to automatically mirror the trading strategies and actions of experienced traders on the platform."},

  {"question": "How can I start following top traders on PrimeTrader?", "answer": "To follow top traders, you can typically browse the list of available traders, review their performance, and choose to follow the ones that align with your trading preferences."},

  {"question": "What determines a trader's Assets Under Management (AUM) on PrimeTrader?", "answer": "A trader's AUM is the total value of assets being managed or traded by that trader's followers. It includes the assets being mirrored by those following the trader."},

  {"question": "How do the NFT levels impact AUM?", "answer": "NFT levels may grant certain privileges or benefits, potentially attracting more followers. This, in turn, can increase a trader's AUM and potentially lead to higher earnings."},

  {"question": "If I'm a top trader, how can I monetize my trading skills on PrimeTrader?", "answer": "Top traders on PrimeTrader can earn by attracting followers who choose to mirror their trading strategies. A percentage of the profits generated by followers may be distributed to the top trader."},

  {"question": "What types of trading games can I participate in on PrimeTrader?", "answer": "PrimeTrader offers various trading competitions, including competitions against the market, peer-to-peer challenges, and other formats. The specific types of games can vary."},

  {"question": "How does the Trader vs. Market game work?", "answer": "In the Trader vs. Market game, traders compete against the market's performance. Traders aim to outperform the market within a specified time frame to win rewards."},

  {"question": "Can I challenge other traders directly?", "answer": "Yes, PrimeTrader typically allows traders to challenge each other directly in trading competitions. These challenges can be initiated within the platform."},

  {"question": "How often are trading competitions held, and how can I participate?", "answer": "The frequency of trading competitions can vary, but PrimeTrader generally hosts them regularly. To participate, you can usually enter the competition through the platform, following the provided instructions."},

  {"question": "What are the rewards for winning a trading competition?", "answer": "The rewards for winning a trading competition may include PTT tokens, NFTs, and other digital assets, depending on the competition's format and rules."},

  {"question": "How can I mint, purchase, or trade NFTs on PrimeTrader?", "answer": "The process of minting, purchasing, or trading NFTs on PrimeTrader can typically be done through the platform's NFT marketplace. Detailed instructions are provided on the platform."},

  {"question": "What benefits do different NFT levels offer?", "answer": "Different NFT levels may provide various benefits, such as exclusive access to trading features, reduced fees, or enhanced visibility on the platform. The specific benefits associated with each level will be outlined on PrimeTrader."},

  {"question": "How does the staking and reward system work for PTT tokens?", "answer": "Staking PTT tokens may grant access to different NFT levels, which, in turn, can provide rewards, privileges, and potential earnings within the PrimeTrader ecosystem. Details can be found within the platform."},

  {"question": "How can I connect with other traders or join the PrimeTrader community?", "answer": "You can typically engage with other traders and the PrimeTrader community through social features within the platform, such as chat rooms, forums, and social media channels."},

  {"question": "Are there any educational resources available to help me become a better trader?", "answer": "PrimeTrader often provides educational resources, including tutorials, articles, and videos, to help users improve their trading skills and knowledge."},
{"question":" Are there any ongoing promotions or collaborations I should be aware of?", "answer":" PrimeTrader may run promotions and collaborations with other entities in the crypto and trading space. These will be announced on the platform and through official communication channels."},
 {"question": "Hello", "answer": "Hi there!"},
  {"question": "Hi", "answer": "Hello! How can I help you today?"},
  {"question": "Hey", "answer": "Hey! What's up?"},
  {"question": "Good morning", "answer": "Good morning!"},
  {"question": "Good afternoon", "answer": "Good afternoon!"},
  {"question": "Good evening", "answer": "Good evening!"},
  {"question": "How are you?", "answer": "I'm doing well, thank you! How about you?"},
  {"question": "What's up?", "answer": "Not much, just here to assist you."},
  {"question": "Nice to meet you", "answer": "Nice to meet you too!"},
  {"question": "Greetings", "answer": "Greetings! How may I be of service?"},
  {"question": "Hey there", "answer": "Hello! How can I assist you today?"},
  {"question": "Yo", "answer": "Yo! What's going on?"},
  {"question": "Good to see you", "answer": "Good to see you too!"},
  {"question": "Howdy", "answer": "Howdy! What brings you here?"},
  {"question": "Hola", "answer": "Hola! ¿En qué puedo ayudarte hoy? (Hello! How can I help you today?)"},
  {"question": "Greetings and salutations", "answer": "Greetings and salutations! How can I be of assistance?"},
  {"question": "What's cracking?", "answer": "Not much, just ready to help!"},
  {"question": "Hiya", "answer": "Hiya! What can I do for you today?"},
  {"question": "Bonjour", "answer": "Bonjour! Comment puis-je vous aider aujourd'hui? (Hello! How can I help you today?)"},
  {"question": "Sup?", "answer": "Hey! Not much, just here to assist."},
  {"question": "How's it going?", "answer": "It's going well, thank you! How about you?"},
  {"question": "Salutations", "answer": "Salutations! How may I assist you?"},
  {"question": "Namaste", "answer": "Namaste! How can I assist you today?"},
  {"question": "Hello, friend", "answer": "Hello! I'm here to help, my friend."},
  {"question": "Hi there", "answer": "Hello! What can I do for you today?"}
]

@app.route('/ask', methods=['POST'])
def ask_question():
    try:

        new_question = request.json['question']

        if new_question.lower() == "bye":
            return jsonify({"response": "Bot: Bye! See you again shortly."})

        new_question_embedding = model.encode([new_question])

        similarities = cosine_similarity(new_question_embedding, loaded_embedding_array)
        most_similar_index = np.argmax(similarities)
        most_similar_sentence = sentences[most_similar_index]

        if similarities[0][most_similar_index] > 0.50:
            response = most_similar_sentence['answer']
        else:
            response = "I'm sorry for any confusion. Could you please rephrase your request or ask in a different way? I'm here to help!"

        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=1244)





































# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# from transformers import pipeline
# from sentence_encoder import model, sentences


# loaded_embedding_array = np.load("sentence_embeddings_minilm.npy")

# new_question = ""
# if(new_question == "bye"):
#   print(f"Bot: Bye See you Again shortly.")

# new_question_embedding = model.encode([new_question])
# similarities = cosine_similarity(new_question_embedding, loaded_embedding_array)
# most_similar_index = np.argmax(similarities)
# most_similar_sentence = sentences[most_similar_index]

# if(similarities[0][most_similar_index] > 0.50):
#   print(most_similar_sentence['answer'])
# else:
#   print(f"I'm sorry for any confusion. Could you please rephrase your request or ask in a different way? I'm here to help!")
