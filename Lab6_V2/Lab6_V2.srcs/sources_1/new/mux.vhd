library ieee;
use ieee.std_logic_1164.ALL;


entity  mux is 
port (
        val : in std_logic;
        sel : in std_logic_vector (3 downto 0);
        lines : out std_logic_vector (15 downto 0)
        );
end entity;

architecture behav of mux is


begin
process (sel, val )
begin
case sel is
    when "0000" =>
        lines <=  "ZZZZZZZZZZZZZZZ"&val ;
    when "0001" =>
        lines <= "ZZZZZZZZZZZZZZ"&val&"Z";
    when "0010" =>
        lines <= "ZZZZZZZZZZZZZ"&val&"ZZ";
    when "0011" =>
        lines <= "ZZZZZZZZZZZZ"&val&"ZZZ";
    when "0100" =>
        lines <= "ZZZZZZZZZZZ"&val&"ZZZZ";
    when "0101" =>
        lines <= "ZZZZZZZZZZ"&val&"ZZZZZ";
    when "0110" =>
        lines <= "ZZZZZZZZZ"&val&"ZZZZZZ";
    when "0111" =>
        lines <= "ZZZZZZZZ"&val&"ZZZZZZZ";
    when "1000" =>
        lines <= "ZZZZZZZ"&val&"ZZZZZZZZ";
    when "1001" =>
        lines <= "ZZZZZZ"&val&"ZZZZZZZZZ";
    when "1010" =>
        lines <= "ZZZZZ"&val&"ZZZZZZZZZZ";
    when "1011" =>
        lines <= "ZZZZ"&val&"ZZZZZZZZZZZ";
--    when "1100" =>
--        lines(12) <= val;
--    when "1101" =>
--        lines(13) <= val;
--    when "1110" =>
--        lines(14) <= val;
--    when "1111" =>
--        lines(15) <= val;
    when others =>
        lines(15) <= val;
end case;
end process ;
end behav;