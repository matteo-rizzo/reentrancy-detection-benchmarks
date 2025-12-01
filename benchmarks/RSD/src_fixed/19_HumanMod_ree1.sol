pragma solidity ^0.8.0;

// SPDX-License-Identifier: GPL-3.0
contract C {
    mapping (address => uint256) public balances;
    bool private flag;

    modifier nonReentrant() {
        // require(!flag, "Locked");
        flag = true;
        _;
        flag = false;
    }
    modifier isHuman() {
        address _addr = msg.sender;
        uint256 _codeLength;
        assembly {_codeLength := extcodesize(_addr)}
        require(_codeLength == 0, "sorry humans only");
        _;
    }

    function transfer(address to) isHuman() nonReentrant public {
        require(balances[msg.sender] > 0, "Insufficient funds");
        (bool success, ) = to.call{value:balances[msg.sender]}("");
        require(success, "Call failed");
        balances[msg.sender] = 0;
    }

    
    function deposit() isHuman() public payable {
        balances[msg.sender] += msg.value;       
    }

}