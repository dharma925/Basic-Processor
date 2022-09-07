library ieee;
use ieee.std_logic_1164.ALL;
use ieee.std_logic_unsigned.all;

entity reg is
    port(
        din : in std_logic ;
        clk: in std_logic ;
        rst: in std_logic ;
        mode : in std_logic ;
        input : in std_logic_vector (15 downto 0);
        output : out std_logic_vector (15 downto 0);
        output_alu : out std_logic_vector (15 downto 0)
    );
end entity;

architecture  behav of reg is
    signal t1 : std_logic ;
    signal stored : std_logic_vector (15 downto 0) := (others=>'Z');
begin
    process(clk,rst,mode)
    begin
        if (din = '1' and input /= "ZZZZZZZZZZZZZZZZ") then
            stored <= input;
        end if;
        if rst = '1' then
            stored <= (others=>'Z');
        elsif rising_edge (clk) and mode='0' and din='0' then
            stored <= input ;
        end if;
    end process;
    output <= stored when mode ='1' else (others=>'Z');
    output_alu  <= stored;
end behav;
