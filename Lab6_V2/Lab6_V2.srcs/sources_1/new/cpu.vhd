library ieee;
use ieee.std_logic_1164.all;

entity cpu is 
    Port(
         reset : in std_logic ;
         cpuclk : in std_logic ;
         mdin : in std_logic_vector (15 downto 0);
         outputtt : out std_logic_vector(15 downto 0) 
            );
end entity ; 

architecture structural of cpu is

component   mux is 
port (
        val : in std_logic;
        sel : in std_logic_vector (3 downto 0);
        lines : out std_logic_vector (15 downto 0)
        );
end component ;

component  alu is
port(
        en: in std_logic ;
        a: in std_logic_vector (15 downto 0);
        b: in std_logic_vector (15 downto 0);
        op : in std_logic_vector (2 downto 0);
        output : out std_logic_vector (15 downto 0)
        );
end component ;

component  cu is
  Port (
        reset : in std_logic ;
        clk : in std_logic ;
        ir : in std_logic_vector (8 downto 0);
        muxval: out std_logic ;
        muxaddr : out std_logic_vector (3 downto 0);
        aluop : out std_logic_vector (2 downto 0);
        aluen : out std_logic
     );
end component ;

component  reg is 
port(
    din : in std_logic ;
    clk: in std_logic ;
    rst: in std_logic ;
    mode : in std_logic ;
    input : in std_logic_vector (15 downto 0);
    output : out std_logic_vector (15 downto 0);
    output_alu : out std_logic_vector (15 downto 0)
    );
end component ;


--Signals
signal sbusin,sbusout,sir,slines : std_logic_vector (15 downto 0); 
signal sclk : std_logic ;
signal srst, smuxval : std_logic ; 
signal alua,aluo : std_logic_vector (15 downto 0);
signal smuxaddr  : std_logic_vector (3 downto 0); 
signal opcode : std_logic_vector (2 downto 0);
signal saluen : std_logic ;

begin
Buss: reg port map (din =>'1',clk => sclk,rst => srst,mode =>'U',input => sbusout, output_alu => sbusin); 
R0: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(0),input => sbusin, output => sbusout);
R1: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(1),input => sbusin, output => sbusout);
R2: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(2),input => sbusin, output => sbusout);
R3: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(3),input => sbusin, output => sbusout);
R4: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(4),input => sbusin, output => sbusout);
R5: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(5),input => sbusin, output => sbusout);
R6: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines (6),input => sbusin, output => sbusout);
R7: reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(7),input => sbusin, output => sbusout);
DINN: reg port map (din => '1',clk => sclk,rst => srst,mode =>slines(8),input => mdin, output => sbusout);
A : reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(9),input => sbusin, output_alu => alua);
G : reg port map (din => '1',clk => sclk,rst => srst,mode =>slines(10),input => aluo, output => sbusout);
IR : reg port map (din => '0',clk => sclk,rst => srst,mode =>slines(11),input => sbusin, output_alu => sir);

demux : mux port map (val =>smuxval ,sel =>smuxaddr , lines => slines );
control : cu port map (reset => srst , clk => sclk  , ir => sir(8 downto 0), muxval => smuxval , muxaddr => smuxaddr , aluop => opcode, aluen => saluen);
aluu : alu  port map(en => saluen,a => alua,b => sbusin ,op => opcode,output=>aluo);

sclk <= cpuclk;
srst <= reset ;

outputtt <= sbusout  ;
end structural;