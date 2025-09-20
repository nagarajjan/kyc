# kyc
# Agentic RAG KYC

# Run the script:
sh
# PIP 
pip install langchain langchain_community langgraph chromadb

# python your_script_name.py

Observe the execution: The output will show the flow of control through the LangGraph state machine.

Orchestrator Decision: You will see the orchestrator's output, which contains its decision about which tool to call next and the specific query to use. For example: ---Orchestrator Decision: internal_kyc_tool: beneficial owners of ABC Inc.---

Agent Execution: The run_agent function will print a message indicating which tool it is running. For example: ---Running agent 'internal_kyc_tool' with input: 'beneficial owners of ABC Inc.'---

Agent Output: The tool's output will be passed back into the graph's state. Since this is a streaming output from LangGraph, each node's output will be printed as it occurs.

Final State: The graph will terminate once a node returns to the END state.


